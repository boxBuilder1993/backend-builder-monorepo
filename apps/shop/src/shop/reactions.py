"""Domain reactions. They never see HTTP shapes."""

import logging

from backend_builder import Context, Reaction

from shop.entities import Order, OrderLine

UNPAID_EXPIRY = "1 hour"
logger = logging.getLogger(__name__)


class OrderTotal(Reaction):
    """Deferred: runs after any order line is created, repriced or deleted.
    Input is the change record; current state is loaded fresh."""

    async def calculate(self, ctx: Context) -> None:
        line = await ctx.get_by_id(OrderLine, ctx.input["id"], include_deleted=True)
        if line is None:
            raise LookupError(f"no order line {ctx.input['id']}")
        order = await ctx.get_by_id(Order, line.order_id)
        if order is None:
            return  # the order itself is gone; nothing to total
        lines = await ctx.query(
            OrderLine,
            "select * from order_line where order_id = :order_id and deleted_at is null",
            order_id=order.id,
        )
        order.total = sum(item.price * item.qty for item in lines)


class CreateOrder(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order = ctx.create(Order, user_id=ctx.input["user_id"], tenant=ctx.input["tenant"])
        for line in ctx.input["lines"]:
            ctx.create(OrderLine, order_id=order.id, price=line["price"], qty=line["qty"])
        ctx.save("order", order)


class UpdateLine(Reaction):
    """Reprice or requantify a line. The total follows through OnChange."""

    async def calculate(self, ctx: Context) -> None:
        line = await ctx.get_by_id(OrderLine, ctx.input["line_id"])
        if line is None or line.order_id != ctx.input["order_id"]:
            raise LookupError(f"no line {ctx.input['line_id']} on order {ctx.input['order_id']}")
        if "price" in ctx.input:
            line.price = ctx.input["price"]
        if "qty" in ctx.input:
            line.qty = ctx.input["qty"]
        ctx.save("line", line)


class DeleteLine(Reaction):
    """Soft-delete a line. The total follows through OnDelete."""

    async def calculate(self, ctx: Context) -> None:
        line = await ctx.get_by_id(OrderLine, ctx.input["line_id"])
        if line is None or line.order_id != ctx.input["order_id"]:
            raise LookupError(f"no line {ctx.input['line_id']} on order {ctx.input['order_id']}")
        ctx.delete(line)
        ctx.save("line", line)


class ExpireUnpaidOrders(Reaction):
    """Scheduled: every minute, expire orders still ``new`` after the expiry window."""

    async def calculate(self, ctx: Context) -> None:
        stale = await ctx.query(
            Order,
            f"select * from \"order\" where status = 'new' and deleted_at is null "
            f"and created_at < now() - interval '{UNPAID_EXPIRY}'",
        )
        for order in stale:
            order.status = "expired"
        ctx.save("expired", [order.id for order in stale])


class NotifyOrder(Reaction):
    """Deferred: tell the notification provider about a new order. The
    provider's key comes from ``SECRET_NOTIFY_API_KEY``; what we would send
    is saved so the run record shows the redaction at work."""

    async def calculate(self, ctx: Context) -> None:
        order = await ctx.get_by_id(Order, ctx.input["id"])
        if order is None:
            return
        key = ctx.secret("notify_api_key")
        logger.info("notify requested order=%s tenant=%s", order.id, order.tenant)
        ctx.save("request", {"authorization": f"Bearer {key}", "order": order.id})
