"""HTTP surface: request and response models, API reactions, builders, endpoints.

API reactions translate the structured request into domain inputs. Builders
read the root context after the run and shape the response.
"""

from backend_builder import Context, Endpoint, Reaction
from pydantic import BaseModel, ConfigDict

from shop.entities import Order, OrderLine
from shop.reactions import CreateOrder

# --- request models -----------------------------------------------------------


class LineIn(BaseModel):
    price: int
    qty: int


class CreateOrderBody(BaseModel):
    user_id: str
    lines: list[LineIn]


class TenantHeaders(BaseModel):
    x_tenant_id: str = "default"


class OrderPath(BaseModel):
    order_id: str


class GetOrderQuery(BaseModel):
    include_lines: bool = False


# --- response models ----------------------------------------------------------


class LineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    price: int
    qty: int


class OrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    user_id: str
    tenant: str
    total: int


class OrderResponse(BaseModel):
    order: OrderOut
    lines: list[LineOut]


# --- API reactions ------------------------------------------------------------


class CreateOrderApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        body = ctx.input["body"]
        await ctx.trigger(
            CreateOrder,
            {
                "user_id": body["user_id"],
                "lines": body["lines"],
                "tenant": ctx.input["headers"]["x_tenant_id"],
            },
        )
        ctx.save("include_lines", True)


class GetOrderApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        order_id = ctx.input["path"]["order_id"]
        order = ctx.get_by_id(Order, order_id)
        if order is None:
            raise LookupError(f"no order {order_id}")
        ctx.save("order", order)
        ctx.save("include_lines", ctx.input["query"]["include_lines"])


# --- builders -----------------------------------------------------------------


def order_response(ctx: Context) -> OrderResponse:
    order: Order = ctx.load("order")
    lines: list[OrderLine] = []
    if ctx.load("include_lines"):
        lines = ctx.get(OrderLine, lambda line: line.order_id == order.id)
    return OrderResponse(
        order=OrderOut.model_validate(order),
        lines=[LineOut.model_validate(line) for line in lines],
    )


endpoints = [
    Endpoint(
        method="POST",
        path="/orders",
        reaction=CreateOrderApi,
        body=CreateOrderBody,
        headers=TenantHeaders,
        response=OrderResponse,
        build=order_response,
    ),
    Endpoint(
        method="GET",
        path="/orders/{order_id}",
        reaction=GetOrderApi,
        path_model=OrderPath,
        query=GetOrderQuery,
        response=OrderResponse,
        build=order_response,
    ),
]
