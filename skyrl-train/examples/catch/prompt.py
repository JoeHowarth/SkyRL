SYSTEM_PROMPT = (
    "You are playing a catch game. The ball falls one cell per turn. "
    "The paddle is on the bottom row. "
    "Respond with exactly one token: L (left), R (right), or S (stay)."
)

STATE_TEMPLATE = (
    "State: grid_w={grid_w}, grid_h={grid_h}, "
    "ball=({ball_x},{ball_y}), paddle_x={paddle_x}, turns_left={turns_left}.\n"
    "Action (L/R/S):"
)


def format_state(*, grid_w: int, grid_h: int, ball_x: int, ball_y: int, paddle_x: int, turns_left: int) -> str:
    return STATE_TEMPLATE.format(
        grid_w=grid_w,
        grid_h=grid_h,
        ball_x=ball_x,
        ball_y=ball_y,
        paddle_x=paddle_x,
        turns_left=turns_left,
    )
