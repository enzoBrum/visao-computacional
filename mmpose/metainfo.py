
dataset_info = dict(
    dataset_name="mpd-rpg1k",
    keypoint_info={
        0: dict(name="A", id=0, color=[51, 153, 255], type="upper", swap="B"),
        1: dict(name="B", id=1, color=[51, 153, 255], type="upper", swap="A"),
    },
    skeleton_info={
        0: dict(link=("A", "B"), id=0, color=[51, 153, 255]),
    },
    joint_weights=[1.0, 1.0],
    sigmas=[
        0.025,
        0.025,
    ],
)
