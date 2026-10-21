"""Domain reactions. They never see HTTP shapes."""

from backend_builder import Context, Reaction

from shop.entities import Order, OrderLine


class OrderTotal(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order = ctx.get_by_id(Order, ctx.input["order_id"])
        if order is None:
            raise LookupError(f"no order {ctx.input['order_id']}")
        lines = ctx.get(OrderLine, lambda line: line.order_id == order.id)
        order.total = sum(line.price * line.qty for line in lines)


class CreateOrder(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order = ctx.create(Order, user_id=ctx.input["user_id"], tenant=ctx.input["tenant"])
        for line in ctx.input["lines"]:
            ctx.create(OrderLine, order_id=order.id, price=line["price"], qty=line["qty"])
        await ctx.trigger(OrderTotal, {"order_id": order.id})
        ctx.save("order", order)
