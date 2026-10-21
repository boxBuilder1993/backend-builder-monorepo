from backend_builder import Context, Reaction

from fixture_app.entities import Widget


class Bump(Reaction):
    async def calculate(self, ctx: Context) -> None:
        widget = ctx.get_by_id(Widget, ctx.input["widget_id"])
        assert widget is not None
        widget.count += 1


class MakeWidget(Reaction):
    async def calculate(self, ctx: Context) -> None:
        widget = ctx.create(Widget, name=ctx.input["name"])
        ctx.save("widget", widget)
