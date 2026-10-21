"""Domain reactions. They never see HTTP shapes."""

from backend_builder import Context, Reaction

from shop.entities import Order, OrderLine


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
