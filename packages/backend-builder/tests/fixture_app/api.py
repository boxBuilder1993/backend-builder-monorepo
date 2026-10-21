from backend_builder import Context, Endpoint, Reaction
from pydantic import BaseModel, ConfigDict

from fixture_app.entities import Widget
from fixture_app.reactions import MakeWidget


class MakeWidgetBody(BaseModel):
    name: str


class WidgetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    count: int


class MakeWidgetApi(Reaction):
    async def calculate(self, ctx: Context) -> None:
        await ctx.trigger(MakeWidget, {"name": ctx.input["body"]["name"]})


async def widget_response(ctx: Context) -> WidgetOut:
    return WidgetOut.model_validate(ctx.load("widget"))


endpoints = [
    Endpoint(
        method="POST",
        path="/widgets",
        reaction=MakeWidgetApi,
        body=MakeWidgetBody,
        response=WidgetOut,
        build=widget_response,
    ),
]

__all__ = ["Widget", "endpoints"]
