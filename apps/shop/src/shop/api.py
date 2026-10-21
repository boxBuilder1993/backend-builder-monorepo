"""HTTP surface: request and response models, API reactions, builders, endpoints.

API reactions translate the structured request into domain inputs. Builders
read the root context after the run and shape the response.
"""

from backend_builder import Context, Endpoint, Reaction, Reply
from pydantic import BaseModel, ConfigDict

from shop.entities import Order, OrderLine
from shop.reactions import CreateOrder, DeleteLine, UpdateLine

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


class LinePath(BaseModel):
    order_id: str
    line_id: str


class UpdateLineBody(BaseModel):
    price: int | None = None
    qty: int | None = None


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
    status: str


class LineResponse(BaseModel):
    line: LineOut
    deleted: bool = False


class OrderResponse(BaseModel):
    order: OrderOut
    lines: list[LineOut]


class ErrorOut(BaseModel):
    error: str
    message: str


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
        order = await ctx.get_by_id(Order, order_id)
        if order is None:
            # An outcome the run understood, not a failure: the builder replies 404.
            ctx.save("not_found", {"error": "NotFound", "message": f"no order {order_id}"})
            return
        ctx.save("order", order)
        ctx.save("include_lines", ctx.input["query"]["include_lines"])


class UpdateLineApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        path = ctx.input["path"]
        body = {k: v for k, v in ctx.input["body"].items() if v is not None}
        await ctx.trigger(
            UpdateLine, {"order_id": path["order_id"], "line_id": path["line_id"], **body}
        )


class DeleteLineApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        path = ctx.input["path"]
        await ctx.trigger(DeleteLine, {"order_id": path["order_id"], "line_id": path["line_id"]})


# --- builders -----------------------------------------------------------------


async def order_response(ctx: Context) -> OrderResponse | Reply:
    if "not_found" in ctx.vars:
        return Reply(status=404, body=ErrorOut(**ctx.load("not_found")))
    order: Order = ctx.load("order")
    lines: list[OrderLine] = []
    if ctx.load("include_lines"):
        lines = await ctx.query(
            OrderLine,
            "select * from order_line where order_id = :order_id and deleted_at is null",
            order_id=order.id,
        )
    return OrderResponse(
        order=OrderOut.model_validate(order),
        lines=[LineOut.model_validate(line) for line in lines],
    )


async def line_response(ctx: Context) -> LineResponse:
    line: OrderLine = ctx.load("line")
    return LineResponse(line=LineOut.model_validate(line), deleted=line.deleted_at is not None)


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
        responses={404: ErrorOut},
        build=order_response,
    ),
    Endpoint(
        method="PATCH",
        path="/orders/{order_id}/lines/{line_id}",
        reaction=UpdateLineApi,
        path_model=LinePath,
        body=UpdateLineBody,
        response=LineResponse,
        build=line_response,
    ),
    Endpoint(
        method="DELETE",
        path="/orders/{order_id}/lines/{line_id}",
        reaction=DeleteLineApi,
        path_model=LinePath,
        response=LineResponse,
        build=line_response,
    ),
]
