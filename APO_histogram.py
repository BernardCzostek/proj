import PySimpleGUI as sg
import numpy as np


def draw_histogram_graph(graph, img_lut, img_stats, hist_params):

    graph.erase()

    # main
    CANVAS_SIZE = DATA_SIZE = (621, 371)
    CANVAS_BORDER_WIDTH = 1
    DATA_HEIGHT_MAX = 300
    FRAME_WIDTH = 1
    GRAPH_WIDTH = 350
    GRAPH_HEIGHT = DATA_HEIGHT_MAX

    # OFFSETS
    GRAPH_OFFSET_Y = 5
    GRAPH_OFFSET_X = 0

    MARGIN_X = ((CANVAS_SIZE[0] - GRAPH_WIDTH) ) + GRAPH_OFFSET_X
    MARGIN_Y = ((CANVAS_SIZE[1] - GRAPH_HEIGHT) ) + GRAPH_OFFSET_Y

    # graph handles
    GRAPH_TOP_LEFT = (MARGIN_X, GRAPH_HEIGHT + MARGIN_Y)
    GRAPH_BOTTOM_RIGHT = (GRAPH_WIDTH + MARGIN_X, MARGIN_Y)

    # x-axis tick label offset
    TICK_LABEL_OFFSET_X = 8

    # x-axis label
    LABEL_X = hist_params["x-axis_label"]
    LABEL_X_POS_X = MARGIN_X
    LABEL_X_POS_Y = MARGIN_Y

    # y-axis tick and tick label offset
    TICK_LENGTH_Y = 4
    TICK_LABEL_OFFSET_Y = 25

    # bars
    BAR_SPACING = 2
    BAR_WIDTH = 2
    BAR_START = MARGIN_X + FRAME_WIDTH
    BAR_COLOR = hist_params["bar_color"]
    HIGHLIGHT_BAR_COLOR = hist_params["highlight_bar_color"]

    # HISTOGRAM MODE
    HISTOGRAM_MODE = img_stats["mode"]

    # clear graph
    graph.erase()

    # draw canvas frame
    graph.draw_rectangle(
        top_left=(0, CANVAS_SIZE[1] - CANVAS_BORDER_WIDTH),
        bottom_right=(CANVAS_SIZE[0] - CANVAS_BORDER_WIDTH - 1, 1),
        line_color="black",
        line_width=CANVAS_BORDER_WIDTH,
    )

    # draw histogram frame
    graph.draw_rectangle(
        top_left=GRAPH_TOP_LEFT,
        bottom_right=GRAPH_BOTTOM_RIGHT,
        line_color="black",
        line_width=FRAME_WIDTH,
    )

    # draw x-axis label
    graph.draw_text(
        text=LABEL_X,
        location=(LABEL_X_POS_X, LABEL_X_POS_Y),
        font="_ 12",
    )

    # draw x-axis tick labels
    graph.draw_text(
        text="0",
        location=(MARGIN_X, MARGIN_Y),
        font="_ 8",
    )
    graph.draw_text(
        text="255",
        location=(
            MARGIN_X + GRAPH_WIDTH,
            MARGIN_Y - TICK_LABEL_OFFSET_X,
        ),
        font="_ 8",
    )

    # draw tick
    line_from = (
        GRAPH_TOP_LEFT[0] - TICK_LENGTH_Y,
        GRAPH_TOP_LEFT[1],
    )
    line_to = (GRAPH_TOP_LEFT[0], GRAPH_TOP_LEFT[1])
    graph.draw_line(point_from=line_from, point_to=line_to, color="black", width=1)

    # draw tick label
    tick_label = HISTOGRAM_MODE
    if tick_label > 1e6:
        tick_label = "{:.2e}".format(tick_label)
    else:
        tick_label = int(tick_label)

    graph.draw_text(
        text=tick_label,
        location=(line_from[0] - TICK_LABEL_OFFSET_Y, line_from[1]),
        font="_ 8",
    )

    # control figure for id
    id = graph.draw_text(
        text="",
        location=(0, 0),
        font="_ 0",
    )
    
    fig_num = 7
    graph.metadata["last_id"] = id - 256 * 3 - fig_num


def new_histogram(img_type, img_lut, img_stats, hist_params):

    WINDOW_SIZE = (680, 500)
    CANVAS_SIZE = DATA_SIZE = (621, 371)

    # image type
    isRGB = True if img_type == "RGB" else False

    # define window
    window_graph = sg.Window(
        title="Histogram",
        layout=[
            [
                sg.Graph(
                    canvas_size=CANVAS_SIZE,
                    graph_bottom_left=(0, 0),
                    graph_top_right=DATA_SIZE,
                    background_color="#FFF",
                    key="-HISTGRAPH-",
                    drag_submits=True,
                    enable_events=False,
                    motion_events=True,
                    metadata={},
                )
            ],
            [
                sg.Column(
                    [
                        [
                            sg.Text(
                                "Value: ---",
                                background_color="#FFF",
                                size=15,
                                key="-HISTVAL-",
                            ),
                            sg.Text(
                                f"N: {round(img_stats['N'])}",
                                background_color="#FFF",
                                size=15,
                                key="-HISTN-",
                            ),
                            sg.Text(
                                f"Min: {img_stats['bounds'][0]}",
                                background_color="#FFF",
                                size=15,
                                key="-HISTMIN-",
                            ),
                            sg.Text(
                                f"Max: {img_stats['bounds'][1]}",
                                background_color="#FFF",
                                size=15,
                                key="-HISTMAX-",
                            ),
                        ],
                        [
                            sg.Text(
                                "Count: ---",
                                background_color="#FFF",
                                size=15,
                                key="-HISTCOUNT-",
                            ),
                            sg.Text(
                                f"StdDev: {round(img_stats['stdev'],3)}",
                                background_color="#FFF",
                                size=15,
                                key="-HISTSTDEV-",
                            ),
                            sg.Text(
                                f"Mode: {img_stats['mode_position']} ({round(img_stats['mode'])})",
                                background_color="#FFF",
                                size=15,
                                key="-HISTMODE-",
                            ),
                            sg.Text(
                                f"Mean: {round(img_stats['mean'],3)}",
                                background_color="#FFF",
                                size=15,
                                key="-HISTMEAN-",
                            ),
                        ],
                    ],
                    background_color="#FFF",
                    pad=(0, 10),
                )
            ],
            # [sg.Text(f"", background_color="#FFF", pad=(0, 10))],
            [sg.HorizontalSeparator(color="black", pad=0)],
            [   
                sg.Button("", visible=False),  # to remove focus
                sg.Button(
                    "RGB", key="-HISTRGB-", border_width=0, visible=isRGB, metadata=0
                ),
                sg.Button("Copy", key="-HISTCOPY-", border_width=0),
                sg.Button("Close", key="Exit", border_width=0),
            ],
        ],
        finalize=False,
        size=WINDOW_SIZE,
        element_justification="center",
        background_color="#FFF",
        keep_on_top=True,
        metadata={"active_lut": {"lut": img_lut, "stats": img_stats, "hist_params": hist_params}},
    )
    
    draw_histogram_graph(window_graph["-HISTGRAPH-"], img_lut, img_stats, hist_params)

    return window_graph
