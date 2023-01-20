import cv2 as cv
import PySimpleGUI as sg
import os
import numpy as np

import APO_cvt_png
import init
import config


######################################
########     MAIN WINDOW
######################################


def get_main_window():
    return init.main_window


def is_main_window(window):
    return bool(window == init.main_window)


def get_main_window_metadata():
    return get_main_window().metadata


def get_window_count():
    return get_main_window_metadata()["window_count"]


def get_open_windows():
    return get_main_window_metadata()["open_windows"]


def get_active_window():
    return get_main_window_metadata()["active_window"]


def get_open_windows_labels():
    return [x["label"] for x in get_open_windows().values()]


def get_open_windows_labels_with_id():
    l = []
    open_windows = get_open_windows()
    for e in open_windows:
        l.append(f"({e}) {open_windows[e]['label']}")
    return l


def add_to_open_windows(window):
    get_open_windows()[get_window_id(window)] = {
        "label": get_label(window),
        "window": window,
    }


def remove_from_open_windows(window):
    if is_image_window(window):
        get_open_windows().pop(get_window_id(window))


def set_menu_on():
    get_main_window()["-MENUBAR-"].update(config.menu_on)


def set_menu_off(window=None):
    if window:
        main_window = window
    else:
        main_window = get_main_window()

    main_window["-MENUBAR-"].update(config.menu_off)


def set_window_count(value):
    get_main_window_metadata()["window_count"] = value


def set_active_window(window):

    if window is None:
        get_main_window_metadata()["active_window"] = None
        return

    if get_active_window() is not None:
        get_active_window()["-ISACTIVE-"].update("")

    window["-ISACTIVE-"].update("ACTIVE WINDOW")
    get_main_window_metadata()["active_window"] = window


######################################
########     IMAGE WINDOW
######################################


def is_image_window(window):
    # only windows with images are considered windows, rest are popups
    return bool(get_window_id(window))


def get_img_metadata_handle(window):
    return window["-IMAGE-"].metadata


def get_window_id(window):
    if window.metadata and "id" in window.metadata:
        return window.metadata["id"]


def get_window_by_id(id):
    return get_open_windows()[id]["window"]


def get_mat_by_id(id):
    return get_mat(get_window_by_id(id))


def get_scale_list(window):
    return get_img_metadata_handle(window)["scale"]["list"]


def get_scale_state(window):
    return get_img_metadata_handle(window)["scale"]["state"]


def get_mat(window):
    return get_img_metadata_handle(window)["mat"].copy()


def get_label(window):
    return get_img_metadata_handle(window)["label"]


def get_type(window):
    return get_img_metadata_handle(window)["type"]


def get_scaled_mat(scale_percent, mat):

    width = int(mat.shape[1] * scale_percent / 100)
    height = int(mat.shape[0] * scale_percent / 100)
    dim = (width, height)

    # dont scale if image is too small
    if dim[0] < 1 or dim[1] < 0:
        return None

    # set interpolation mode
    # USED INTER_NEAREST as in FIJI
    interpol = cv.INTER_NEAREST

    # To shrink an image, it will generally look best with #INTER_AREA interpolation,
    # to enlarge an image, it will generally look best with c#INTER_CUBIC (slow)
    # or #INTER_LINEAR (faster but still looks OK).
    # interpol = cv.INTER_AREA if scale_percent < 100 else cv.INTER_LINEAR

    # resize image (mat must be numpy uint8-typed array)
    return cv.resize(mat, dim, interpolation=interpol)


def get_scale_cache_mat(window, scale_state):
    if scale_state in get_img_metadata_handle(window)["scale"]["cached"]:
        return get_img_metadata_handle(window)["scale"]["cached"][scale_state][
            "mat"
        ].copy()
    else:
        return None


def get_scale_cache_img_bytes(window, scale_state):

    if scale_state in get_img_metadata_handle(window)["scale"]["cached"]:
        return get_img_metadata_handle(window)["scale"]["cached"][scale_state][
            "img_bytes"
        ]
    else:
        return None


def set_window_id(window, id):
    window.metadata["id"] = id


def set_mat(window, mat):
    get_img_metadata_handle(window)["mat"] = mat


def set_type(window, type):
    get_img_metadata_handle(window)["type"] = type


def set_img_metadata(window, metadata):
    window["-IMAGE-"].metadata = metadata


def set_scale_cache(window, scale_state, mat, img_bytes):
    get_img_metadata_handle(window)["scale"]["cached"][scale_state] = {
        "img_bytes": img_bytes,
        "mat": mat,
    }


def set_scale_state(window, scale_state):
    get_img_metadata_handle(window)["scale"]["state"] = scale_state


def create_window_image(label, mat, main_window):

    # adjust size, convert to bytes
    is_fail, rmat, img_bytes, scale_state, scale_list = preprocess_img(label, mat)

    if is_fail:
        return None

    # define layout
    l_rcm = ["RCM", ["Select", "Save", "Duplicate"]]
    l_pixels = sg.Text(f"{mat.shape[0]}x{mat.shape[1]} pixels")
    l_type = sg.Text(f"{get_img_type(mat)}", key="-LTYPE-")
    l_scale = sg.Text(f"Scale: {scale_list[scale_state]}%", key="-LSCALE-")
    l_active = sg.Text("", key="-ISACTIVE-")
    l_img = sg.Image(
        data=img_bytes,
        key="-IMAGE-",
        metadata={
            "label": label,
            "type": get_img_type(mat),
            "mat": mat,
            "lut": {},
            "scale": {"state": scale_state, "list": scale_list, "cached": {}},
        },
    )
    l_img_col = [
        sg.Column(
            [[l_img]],
            scrollable=True,
            size_subsample_height=1,
            size_subsample_width=1,
            expand_x=True,
            expand_y=True,
            key="-IMGCOLUMN-",
        )
    ]
    btm = sg.Button("", visible=False)  # to remove focus
    l_btn_scale_up = sg.Button("➕", key="-SCALEUP-", border_width=3)
    l_btn_scale_down = sg.Button("➖", key="-SCALEDOWN-", border_width=3)
    l_layout = [
        [l_pixels, l_type, l_scale, l_active],
        [l_img_col],
        [btm, l_btn_scale_up, l_btn_scale_down],
    ]

    # shift window horizontally by random x pixels
    shift_x = np.random.randint(low=-300, high=300, size=1)

    # create new window
    window = sg.Window(
        label,
        l_layout,
        # finalize=True,
        right_click_menu=l_rcm,
        keep_on_top=True,
        resizable=True,
        relative_location=(shift_x, 0),
        metadata={"main_window": main_window},
    )

    # IMAGE AS GRAPH
    # scaling = scale_list[scale_state] / 100
    # h = mat.shape[1] * scaling
    # w = mat.shape[0] * scaling

    # gp = sg.Graph(
    #     canvas_size=(h+2, w+2),
    #     graph_bottom_left=(0, w+1),
    #     graph_top_right=(h+1, 0),
    #     background_color="#FFF",
    #     key="-IMAGEGRAPH-",
    #     drag_submits=True,
    #     enable_events=False,
    #     motion_events=True,
    #     metadata={},
    # )

    # sg.Window(
    #     label,
    #     [[gp]],
    #     finalize=True,
    #     right_click_menu=l_rcm,
    #     keep_on_top=True,
    #     resizable=True,
    #     relative_location=(300, 0),
    # )

    # gp.draw_image(data=img_bytes, location=(1,1))

    # kazde okno ma w metadata swoje id, jak jest niszczone to usuwane z main window metadata po id zniszczonego okna

    # --OPTIONS-- CACHE CURRENT SCALE IMGBYTES
    set_scale_cache(window, scale_state, rmat, img_bytes)

    return window


def finalize_new_window_image(window):
    if window is not None:
        set_window_count(get_window_count() + 1)
        set_window_id(window, get_window_count())
        window.finalize()
        window.TKroot.title(f"({get_window_count()}) {get_label(window)}")
        add_to_open_windows(window)


def preprocess_img(label, mat):

    #### ADJUST SIZE ####

    # get width and height of image
    mat_h, mat_w = mat.shape[:2]

    # calculate image size
    size = mat_h * mat_w

    # max image size defined by PIL.Image.DecompressionBombError
    PIL_MAX = 178956970

    # consider image to be small if image size < SIZE_SMALL
    SIZE_SMALL = 64 * 64

    # list of allowed zoom percentages
    scale_list = np.array([10, 20, 25, 50, 75, 100, 150, 200])

    if size > PIL_MAX:
        # dont accept images that are too big
        if (size * (scale_list[-1] / 100)) > PIL_MAX:
            sg.popup_ok(
                f'Image "{label}" is too big, size cannot exceed 89 478 485 pixels.',
                title="Cannot open selected image",
                keep_on_top=True,
            )
            return True, None, None, None, None

    elif size < SIZE_SMALL:
        # if image is small allow bigger zoom
        scale_list = np.append(scale_list, [500, 1000, 2000, 3200])

    # get screen dimensions
    temp_window = sg.Window("asd")
    screen_w, screen_h = temp_window.get_screen_dimensions()
    temp_window.close()

    # maximum dimensions of window on a given screen size
    MAX_SCREEN_W = screen_w * 0.85
    MAX_SCREEN_H = screen_h * 0.85

    # calculate scale ratio
    ratio = min(MAX_SCREEN_H / mat_h, MAX_SCREEN_W / mat_w)

    # index of biggest scale percentage that will fit on screen
    scale_state = np.where(scale_list < ratio * 100)[0][-1]

    # --OPTIONS-- DONT ZOOM ON IMAGES THAT ARE BIGGER THAN SIZE_SMALL
    # NO ZOOM BUT STILL RESIZES DOWN (if ratio > 1)
    if size > SIZE_SMALL and ratio > 1:
        scale_state = np.where(scale_list == 100)[0][0]

    # get scale percentage via index
    scale_percent = scale_list[scale_state]

    # scale image
    width = int(mat.shape[1] * scale_percent / 100)
    height = int(mat.shape[0] * scale_percent / 100)
    mat = cv.resize(mat, (width, height), interpolation=cv.INTER_NEAREST)

    # convert mat to bytes
    img_bytes = mat_to_bytes(label, mat)

    return (False, mat, img_bytes, scale_state, scale_list)


def mat_to_bytes(label, mat):
    # get image extension
    ext = os.path.splitext(label)[1]

    # encode matrix to bytes
    img_bytes = cv.imencode(ext, mat)[1].tobytes()

    # convert to png (GUI supports png only)
    img_bytes = APO_cvt_png.cvt_bytes(img_bytes)

    return img_bytes


######################################
########     HISTOGRAM
######################################


def get_img_size(lut):
    return sum(lut)


def get_lut_mode(lut):
    return max(lut)


def get_lut_mode_position(lut, mode):
    return np.where(lut == mode)[0][0]


def get_lut_bounds(lut):
    h_bounds = np.where(lut != 0)
    return (h_bounds[0][0], h_bounds[0][-1])


def get_lut_mean(lut):
    return sum(np.multiply(range(0, 256), lut)) / get_img_size(lut)


def get_lut_stdev(lut):
    stdev_sum = 0
    for i, e in enumerate(lut):
        stdev_sum += ((i - get_lut_mean(lut)) ** 2) * e
    return np.sqrt(stdev_sum / (get_img_size(lut) - 1))


def get_img_stats(lut):
    return {
        "N": get_img_size(lut),
        "mode": get_lut_mode(lut),
        "mode_position": get_lut_mode_position(lut, get_lut_mode(lut)),
        "bounds": get_lut_bounds(lut),
        "mean": get_lut_mean(lut),
        "stdev": get_lut_stdev(lut),
    }


######################################
########     INPUT WINDOW
######################################


def get_window_metadata(window):
    return window.metadata


def get_metadata_by_key(window, key):
    return get_window_metadata(window)[key]


def to_float(val):
    try:
        val = float(val)
    except:
        return None

    return val


def to_int(val):
    try:
        val = int(to_float(val))
    except:
        return None

    return val


def get_dict_value(key, dictionary, default_value=None):
    if key in dictionary:
        return dictionary[key]
    else:
        if default_value is not None:
            return default_value
        else:
            raise KeyError(f"key '{key}' not found in dictionary")


def get_value(window, value, info={}):

    to_type = get_dict_value("to_type", info, to_int)
    min = get_dict_value("min", info, 0)
    max = get_dict_value("max", info, 255)
    val_label = get_dict_value("val_label", info, "Value")

    value = to_type(value)

    if value is None or not (max >= value >= min):
        sg.popup_ok(
            f"{val_label} must be a number between {min} and {max}",
            keep_on_top=True,
        )
        window.close()
        cv_destroy_window("preview")
        return None
    return value


def template_slider_one(window, event, values, params):

    prompt_event = get_dict_value("prompt_event", params)

    if event == prompt_event:
        event_key = get_dict_value("event_key", params)
        func = get_dict_value("func", params)

        header_text = get_dict_value("header_text", params, "")

        val_label = get_dict_value("val_label", params, "Value")
        default_val = get_dict_value("default_val", params, 127)
        slider_range = get_dict_value("slider_range", params, (0, 255))
        slider_resolution = get_dict_value("slider_resolution", params, 1)
        to_type = get_dict_value("to_type", params, to_int)

        min = slider_range[0]
        max = slider_range[1]

        enable_checkbox = get_dict_value("enable_checkbox", params, False)
        checkbox_label = get_dict_value("checkbox_label", params, "")

        checkbox_key = f"-{event_key}CHECKPREVIEW-"
        slider_key = f"-{event_key}SLIDERPREVIEWUPDTVAL-"
        input_key = f"-{event_key}INUPDTVAL-"
        ok_key = f"-{event_key}OKAPL-"

        element = {
            "val_label": val_label,
            "to_type": to_type,
            "min": min,
            "max": max,
        }

        element_map = {
            slider_key: element,
            input_key: element,
        }

        # pairs of elements to be synchronzized
        update_map = {
            slider_key: [
                {
                    "element": input_key,
                    "value": slider_key,
                },
            ],
            checkbox_key: [
                {
                    "element": slider_key,
                    "value": input_key,
                },
            ],
        }

        prompt = sg.Window(
            prompt_event,
            [
                [sg.Text(header_text, visible=bool(header_text), font=(None, 12))],
                [
                    sg.vbottom(sg.Text(f"{val_label}: ")),
                    sg.Slider(
                        default_value=default_val,
                        range=slider_range,
                        resolution=slider_resolution,
                        orientation="h",
                        s=(32, 15),
                        key=slider_key,
                        enable_events=True,
                    ),
                    sg.vbottom(sg.Input(default_val, key=input_key, size=(5, 1))),
                ],
                [
                    sg.Checkbox(
                        checkbox_label,
                        key=checkbox_key,
                        enable_events=True,
                        visible=enable_checkbox,
                        pad=(0, 10),
                    )
                ],
                [sg.Button("OK", key=ok_key)],
            ],
            modal=True,
            keep_on_top=True,
            metadata={
                "prompt_event": prompt_event,
                "element_map": element_map,
                "update_map": update_map,
                "function": func,
            },
        )

        transform(prompt, event, values)
    else:
        transform(window, event, values)


def template_slider_two(window, event, values, params):

    prompt_event = get_dict_value("prompt_event", params)

    if event == prompt_event:
        event_key = get_dict_value("event_key", params)
        func = get_dict_value("func", params)

        header_text = get_dict_value("header_text", params, "")
        slider_header = get_dict_value("slider_header", params, "")
        slider_header_2 = get_dict_value("slider_header_2", params, "")

        val_label = get_dict_value("val_label", params, "Low")
        default_val = get_dict_value("default_val", params, 127)
        slider_range = get_dict_value("slider_range", params, (0, 255))
        slider_resolution = get_dict_value("slider_resolution", params, 1)
        to_type = get_dict_value("to_type", params, to_int)

        min = slider_range[0]
        max = slider_range[1]

        val_label_2 = get_dict_value("val_label_2", params, "High")
        default_val_2 = get_dict_value("default_val_2", params, 127)
        slider_range_2 = get_dict_value("slider_range_2", params, (0, 255))
        slider_resolution_2 = get_dict_value("slider_resolution_2", params, 1)
        to_type_2 = get_dict_value("to_type_2", params, to_int)

        min_2 = slider_range_2[0]
        max_2 = slider_range_2[1]

        enable_checkbox = get_dict_value("enable_checkbox", params, False)
        checkbox_label = get_dict_value("checkbox_label", params, "")
        checkbox_key = f"-{event_key}CHECKPREVIEW-"

        slider_key = f"-{event_key}SLIDERPREVIEWUPDTVAL-"
        input_key = f"-{event_key}INUPDTVAL-"

        slider_key_2 = f"-{event_key}SLIDERPREVIEWUPDTVAL_2-"
        input_key_2 = f"-{event_key}INUPDTVAL_2-"

        ok_key = f"-{event_key}OKAPL-"

        element = {
            "val_label": val_label,
            "to_type": to_type,
            "min": min,
            "max": max,
        }

        element_2 = {
            "val_label": val_label_2,
            "to_type": to_type_2,
            "min": min_2,
            "max": max_2,
        }

        element_map = {
            slider_key: element,
            slider_key_2: element_2,
            input_key: element,
            input_key_2: element_2,
        }

        update_map = {
            slider_key: [
                {
                    "element": input_key,
                    "value": slider_key,
                },
                {
                    "value": input_key_2,
                    "element": slider_key_2,
                },
            ],
            slider_key_2: [
                {
                    "element": input_key_2,
                    "value": slider_key_2,
                },
                {
                    "value": input_key,
                    "element": slider_key,
                },
            ],
            checkbox_key: [
                {
                    "element": slider_key,
                    "value": input_key,
                },
                {
                    "element": slider_key_2,
                    "value": input_key_2,
                },
            ],
        }

        prompt = sg.Window(
            prompt_event,
            [
                [sg.Text(header_text, visible=bool(header_text), font=(None, 12))],
                [sg.Text(slider_header, visible=bool(slider_header))],
                [
                    sg.vbottom(sg.Text(f"{val_label}: ")),
                    sg.Slider(
                        default_value=default_val,
                        range=slider_range,
                        resolution=slider_resolution,
                        key=slider_key,
                        orientation="h",
                        s=(32, 15),
                        enable_events=True,
                    ),
                    sg.vbottom(sg.Input(default_val, key=input_key, size=(5, 1))),
                ],
                [sg.Text(slider_header_2, visible=bool(slider_header_2))],
                [
                    sg.vbottom(sg.Text(f"{val_label_2}: ")),
                    sg.Slider(
                        default_value=default_val_2,
                        range=slider_range_2,
                        resolution=slider_resolution_2,
                        key=slider_key_2,
                        orientation="h",
                        s=(32, 15),
                        enable_events=True,
                    ),
                    sg.vbottom(sg.Input(default_val_2, key=input_key_2, size=(5, 1))),
                ],
                [
                    sg.Checkbox(
                        checkbox_label,
                        key=checkbox_key,
                        visible=enable_checkbox,
                        enable_events=True,
                        pad=(0, 20),
                    )
                ],
                [sg.Button("OK", key=ok_key)],
            ],
            keep_on_top=True,
            metadata={
                "prompt_event": prompt_event,
                "element_map": element_map,
                "update_map": update_map,
                "function": func,
            },
        )

        transform(prompt, event, values)
    else:
        transform(window, event, values)


def cv_destroy_window(window_name):
    is_open = cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE)
    if is_open == 1:
        cv.destroyWindow(window_name)


######################################
########     CALCULATIONS
######################################


def calc_lut(mat, calc_type):

    # get height and width
    h, w = mat.shape[:2]

    if calc_type == "GRAYSCALE":

        # create lut table
        lut = np.zeros(256)

        for x in range(0, h):
            for y in range(0, w):
                lut[mat.item(x, y)] += 1

    elif calc_type == "RGB_SEPERATE":
        # RETURNS BGR, NOT RGB
        # img_lut[0] - b
        # img_lut[1] - g
        # img_lut[2] - r

        # set channel to 3
        c = 3

        # create 3 lut tables
        lut = np.zeros(shape=(3, 256))

        for x in range(0, h):
            for y in range(0, w):
                for z in range(0, c):
                    lut[z][mat.item(x, y, z)] += 1

    elif calc_type == "RGB_INTENSITY_WEIGHTED":

        mat = calc_rgb_intensity_weighted(mat)

        lut = calc_lut(mat, "GRAYSCALE")

    return lut


def calc_rgb_intensity_weighted(mat):
    b = mat[:, :, 0] * 0.114
    g = mat[:, :, 1] * 0.587
    r = mat[:, :, 2] * 0.299

    i_mat = b + g + r
    i_mat = np.around(i_mat).astype(np.uint8)
    return i_mat


def calc_cumulative_distribution(mat):

    lut = calc_lut(mat, "GRAYSCALE")

    cumulative_lut = np.zeros(256)

    cumulative_lut[0] = lut[0]
    for i in range(1, 256):
        cumulative_lut[i] = cumulative_lut[i - 1] + lut[i]
        # print(f"cumulative_lut[{i}] = {cumulative_lut[i-1]} + {lut[i]} = {cumulative_lut[i]}")

    return cumulative_lut


######################################
########     GENERAL
######################################


def get_img_type(mat):
    if len(mat.shape) == 2:
        return "GRAYSCALE"
    elif len(mat.shape) == 3:
        return "RGB"
    else:
        raise ("len(mat.shape) is neither 2 or 3")


def apply_lut(mat, lut):

    # get height and width
    h, w = mat.shape

    # adjust value for each pixel as in lut
    for x in range(0, h):
        for y in range(0, w):
            mat.itemset((x, y), lut[mat.item(x, y)])

    return mat


def apply_to_active(func):
    active_window = get_active_window()
    apply_transformation(active_window, func(get_mat(active_window)))


def apply_transformation(window, mat):

    # --OPTIONS-- MODIFY IN PLACE OR CREATE NEW WINDOW
    TRANSFORMATION_MODE = "CREATE_NEW_WINDOW"
    # TRANSFORMATION_MODE = "MODIFY_CURRENT_WINDOW"

    # mockup window for better metadata handling
    new_window = create_window_image(get_label(window), mat, get_main_window())

    if TRANSFORMATION_MODE == "CREATE_NEW_WINDOW":
        finalize_new_window_image(new_window)
    elif TRANSFORMATION_MODE == "MODIFY_CURRENT_WINDOW":

        scale_state = get_scale_state(window)
        scale_percent = get_scale_list(window)[scale_state]
        scaled_mat = get_scaled_mat(scale_percent, mat)
        img_bytes = mat_to_bytes(get_label(window), scaled_mat)

        set_scale_state(new_window, scale_state)
        set_img_metadata(window, get_img_metadata_handle(new_window))
        set_scale_cache(window, scale_state, scaled_mat, img_bytes)

        # update layout
        window["-LTYPE-"].update(f"{get_type(window)}")

        # update image
        window["-IMAGE-"].update(data=img_bytes)


def transform(window, event, values):

    active_window = get_active_window()

    if event == get_metadata_by_key(window, "prompt_event"):
        # show prompt window for transformation input
        window.finalize()

        # # show preview window
        # if active_window:
        #     cv.imshow(
        #         "preview",
        #         get_scale_cache_mat(active_window, get_scale_state(active_window)),
        #     )

    else:

        # convert values to appropriate type
        for v in values:
            if v.find("UPDTVAL") != -1:

                val = get_value(
                    window, values[v], get_metadata_by_key(window, "element_map")[v]
                )
                if val is None:
                    return

                values[v] = val

        # update layout input controls
        if event.find("PREVIEW") != -1:

            for e in get_metadata_by_key(window, "update_map")[event]:

                val = values[e["value"]]

                window[e["element"]].update(val)
                window[e["value"]].update(val)
                values[e["element"]] = val

                # unecessary but ommiting creates inconsistency
                # in values dictionary which may lead to confusion
                values[e["value"]] = val

            # transform cached mat of appropriate scale
            mat = get_metadata_by_key(window, "function")(
                values,
                get_scale_cache_mat(active_window, get_scale_state(active_window)),
            )

            if mat is None:
                return

            # update preview
            cv.imshow("preview", mat)

        elif event.find("OKAPL") != -1:

            # close prompt window for transformation input
            window.close()

            # close preview window
            cv_destroy_window("preview")

            # get mat
            mat = get_metadata_by_key(window, "function")(
                values, get_mat(active_window)
            )
            if mat is None:
                return

            # apply transoformation
            apply_transformation(active_window, mat)


######################################
########     EVENT FUNCTIONS
######################################


def normalize_linear(mat, input_range, output_range):

    min, max = input_range
    low, high = output_range

    lut = np.zeros(256)

    # calculate adjustment value
    adj = (high - low) / (max - min)

    for i in range(0, 256):
        val = (i - min) * adj + low

        if val < 0:
            val = 0
        elif val > 255:
            val = 255

        lut[i] = val

    # round and convert to int
    lut = np.around(lut).astype(int)

    r_mat = apply_lut(mat, lut)

    return r_mat


def normalize_nonlinear(values, mat):

    gamma = values["-NORMHISTNONLININUPDTVAL-"]

    lut = np.zeros(256)

    high = 255

    for i in range(0, 256):
        lut[i] = high * ((i / high) ** (1 / gamma))

    # round and convert to int
    lut = np.around(lut).astype(int)

    r_mat = apply_lut(mat, lut)

    return r_mat


def equalize_histogram(mat):

    # mat = np.array(
    #     [
    #         [52, 55, 61, 66, 70, 61, 64, 73],
    #         [63, 59, 55, 90, 109, 85, 69, 72],
    #         [62, 59, 68, 113, 144, 104, 66, 73],
    #         [63, 58, 71, 122, 154, 106, 70, 69],
    #         [67, 61, 68, 104, 126, 88, 68, 70],
    #         [79, 65, 60, 70, 77, 68, 58, 75],
    #         [85, 71, 64, 59, 55, 61, 65, 83],
    #         [87, 79, 69, 68, 65, 76, 78, 94],
    #     ],
    #     dtype="uint8",
    # )

    cdf_lut = calc_cumulative_distribution(mat)

    h, w = mat.shape[:2]
    cdf_min = cdf_lut[np.where(cdf_lut > 0)[0][0]]
    pixel_num = h * w

    for x in range(0, h):
        for y in range(0, w):
            # print(f"{mat.item(x, y)} = ({cdf_lut[mat.item(x, y)]}-{cdf_min})/({pixel_num}-{cdf_min}) * 255 = {np.around(((cdf_lut[mat.item(x, y)] - cdf_min) / (pixel_num - cdf_min)) * 255)}")
            mat.itemset(
                (x, y),
                np.around(
                    ((cdf_lut[mat.item(x, y)] - cdf_min) / (pixel_num - cdf_min)) * 255
                ),
            )

    return mat


def negation(mat):

    max_intensity = 255

    h, w = mat.shape[:2]

    if len(mat.shape) == 2:
        for x in range(0, h):
            for y in range(0, w):
                mat.itemset((x, y), max_intensity - mat.item(x, y))

    if len(mat.shape) == 3:
        # RETURNS BGR, NOT RGB
        c = 3

        for x in range(0, h):
            for y in range(0, w):
                for z in range(0, c):
                    mat.itemset((x, y, z), max_intensity - mat.item(x, y, z))

    return mat


def threshold_single(values, mat):

    intensity_threshold = values["-THRESHSINGLEINUPDTVAL-"]

    check = values["-THRESHSINGLECHECKPREVIEW-"]

    return calc_threshold_single(mat, intensity_threshold, check)


def img_calc(values, mat):

    math_operations_map = {
        "add": APO_add,
        "subtract (absolute)": APO_subtract_abs,
        "AND": APO_AND,
        "OR": APO_OR,
        "XOR": APO_XOR,
    }
    operand = math_operations_map[values["-IMGCALCSELECTEDOPERATION-"]]

    no_saturation = values["-IMGCALCNOSATURATIONCHECK-"]

    mat = values["-IMGCALCSELECTIMGONEPREVIEW-"]
    mat_2 = values["-IMGCALCSELECTIMGTWOPREVIEW-"]
    
    if values["-IMGCALCEVENTKEY-"].find("IMGCALCSELECTIM") != -1:

        window = values["-IMGCALCWINDOW-"]
        if mat.shape[:2] != mat_2.shape[:2]:
            window["-IMGCALCWARNING-"].update("❌ Images are not of equal size.")
            window["-IMGCALCWARNING-"].update(text_color="red")
            window["-IMGCALCPREVIEW-"].update(disabled=True)
            window["-IMGCALCOKAPL-"].update(disabled=True)
            return None
        else:
            window["-IMGCALCWARNING-"].update("✅ Images are of equal size.")
            window["-IMGCALCWARNING-"].update(text_color="black")
            window["-IMGCALCPREVIEW-"].update(disabled=False)
            window["-IMGCALCOKAPL-"].update(disabled=False)

    if values["-IMGCALCEVENTKEY-"] in ("-IMGCALCOKAPL-", "-IMGCALCPREVIEW-"):

        if get_img_type(mat) == "RGB":
            mat = calc_rgb_intensity_weighted(mat)
        if get_img_type(mat_2) == "RGB":
            mat_2 = calc_rgb_intensity_weighted(mat_2)

        h, w = mat.shape[:2]

        max_intensity = 255

        for x in range(0, h):
            for y in range(0, w):
                # truncating to int (not flooring)
                new_intensity = int(operand(mat.item(x, y), mat_2.item(x, y)))
                if new_intensity > max_intensity:
                    # overflows, set to max_intenisty
                    mat.itemset((x, y), max_intensity)
                else:
                    # no overflow, set to new_intensity
                    mat.itemset((x, y), new_intensity)

        if no_saturation:
            normalize_linear(mat, (np.amin(mat), np.amax(mat)), (0, max_intensity))

        return mat


def calc_math(mat, value, operand, no_saturation):
    

    h, w = mat.shape[:2]

    max_intensity = 255

    for x in range(0, h):
        for y in range(0, w):
            # truncating to int (not flooring)
            new_intensity = int(operand(mat.item(x, y), value))
            if new_intensity > max_intensity:
                # overflows, set to max_intenisty
                mat.itemset((x, y), max_intensity)
            else:
                # no overflow, set to new_intensity
                mat.itemset((x, y), new_intensity)

    if no_saturation:
        normalize_linear(mat, (np.amin(mat), np.amax(mat)), (0, max_intensity))

    return mat


def APO_AND(a, b):
    return a & b


def APO_NOT(a):
    return ~a + 2**8


def APO_negation_NOT(mat):

    h, w = mat.shape[:2]

    if len(mat.shape) == 2:
        for x in range(0, h):
            for y in range(0, w):
                mat.itemset((x, y), APO_NOT(mat.item(x, y)))

    if len(mat.shape) == 3:
        # RETURNS BGR, NOT RGB
        c = 3

        for x in range(0, h):
            for y in range(0, w):
                for z in range(0, c):
                    mat.itemset((x, y, z), APO_NOT(mat.item(x, y, z)))

    return mat


def APO_OR(a, b):
    return a | b


def APO_XOR(a, b):
    return a ^ b


def APO_subtract_abs(a, b):
    return abs(a - b)


def APO_add(a, b):
    return a + b


def APO_multiply(a, b):
    return a * b


def APO_divide(a, b):
    # assume b > 0
    return a / b


def math_add(values, mat):
    return calc_math(
        mat,
        values["-MATHADDINUPDTVAL-"],
        APO_add,
        values["-MATHADDCHECKPREVIEW-"],
    )


def math_multiply(values, mat):
    return calc_math(
        mat,
        values["-MATHMULTIPLYINUPDTVAL-"],
        APO_multiply,
        values["-MATHMULTIPLYCHECKPREVIEW-"],
    )


def math_divide(values, mat):
    return calc_math(
        mat,
        values["-MATHDIVIDEINUPDTVAL-"],
        APO_divide,
        values["-MATHDIVIDECHECKPREVIEW-"],
    )


def calc_threshold_single(mat, intensity_threshold, check):
    # print("actual val: ", intensity_threshold , type(intensity_threshold))
    h, w = mat.shape[:2]

    foreground = 255
    background = 0

    for x in range(0, h):
        for y in range(0, w):
            if mat.item(x, y) < intensity_threshold:
                mat.itemset((x, y), background)
            elif not check:
                mat.itemset((x, y), foreground)

    return mat


def threshold_band(values, mat):

    threshold_band = (
        values["-THRESHBANDINUPDTVAL-"],
        values["-THRESHBANDINUPDTVAL_2-"],
    )

    check = values["-THRESHBANDCHECKPREVIEW-"]

    return calc_threshold_band(mat, threshold_band, check)


def calc_threshold_band(mat, intensity_band, check):

    h, w = mat.shape[:2]

    low, high = intensity_band

    foreground = 255
    background = 0

    for x in range(0, h):
        for y in range(0, w):
            if low < mat.item(x, y) < high:
                mat.itemset((x, y), background)
            elif not check:
                mat.itemset((x, y), foreground)

    return mat


def threshold_adaptive(mat):
    return cv.adaptiveThreshold(
        mat, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2
    )


def threshold_otsu(mat):
    val = cv.threshold(mat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[0]
    print(val)
    sg.popup(f"Threshold value: {val}", keep_on_top=True)
    return cv.threshold(mat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


def line_detection_canny(values, mat):
    threshold_band = (
        values["-LINEDETCANNYINUPDTVAL-"],
        values["-LINEDETCANNYINUPDTVAL_2-"],
    )
    return cv.Canny(mat, threshold_band[0], threshold_band[1])


def line_detection_sobel(mat):

    # apply gaussian blur to remove noise
    img = cv.GaussianBlur(mat, (3, 3), 0)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3, scale=1)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3, scale=1)
    absx = cv.convertScaleAbs(sobelx)
    absy = cv.convertScaleAbs(sobely)
    edge = cv.addWeighted(absx, 0.5, absy, 0.5, 0)
    # cv.imshow('edge Sobel', edge)
    frame_sobel = cv.hconcat((sobelx, sobely))

    # cv.imshow(edge)
    # plt.figure(figsize=(10,10))
    # plt.imshow(frame_sobel, cmap='gray')
    return edge


def line_detection_prewitt(mat):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    img_gaussian = cv.GaussianBlur(mat, (3, 3), 0)
    img_prewittx = cv.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv.filter2D(img_gaussian, -1, kernely)
    cv.imshow("Prewitt X", img_prewittx)
    cv.imshow("Prewitt Y", img_prewitty)

    return mat


def morphology_dilation(mat):
    dilatation_size = 3  # values["-DILATIONINUPDTVAL-"]

    dilation_shape = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(
        dilation_shape,
        (2 * dilatation_size + 1, 2 * dilatation_size + 1),
        (dilatation_size, dilatation_size),
    )

    return cv.dilate(mat, element)


def morphology_erosion(mat):
    erosion_size = 3  # values["-DILATIONINUPDTVAL-"]

    erosion_shape = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(
        erosion_shape,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size),
    )

    return cv.erode(mat, element)


def morphology_erosion(mat):
    erosion_size = 3  # values["-DILATIONINUPDTVAL-"]

    erosion_shape = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(
        erosion_shape,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size),
    )

    return cv.erode(mat, element)


def morphology_opening(mat):
    erosion_size = 3  # values["-DILATIONINUPDTVAL-"]

    erosion_shape = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(
        erosion_shape,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size),
    )

    return cv.morphologyEx(mat, cv.MORPH_OPEN, element)


def morphology_closing(mat):
    erosion_size = 4  # values["-DILATIONINUPDTVAL-"]

    erosion_shape = cv.MORPH_RECT
    element = cv.getStructuringElement(
        erosion_shape,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size),
    )

    return cv.morphologyEx(mat, cv.MORPH_CLOSE, element)
