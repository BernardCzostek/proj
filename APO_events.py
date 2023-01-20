from APO_functions import *
import APO_histogram
import APO_cvt_png


def ev_open(main_window):
    # get file path
    filepaths = sg.popup_get_file(
        "Select one or multiple files",
        no_window=True,
        multiple_files=True,
        file_types=(("Images", ".tif .jpg .png .bmp"),),
    )

    for filepath in filepaths:
        # get file label
        label = os.path.basename(filepath)

        # read image to Matrix
        mat = cv.imread(filepath, cv.IMREAD_UNCHANGED)

        # remove alpha channel
        # if len(img_mat.shape) == 3 and img_mat.shape[2] == 4:
        #     img_mat = cv.cvtColor(img_mat, cv.COLOR_BGRA2BGR)

        # convert BGR to RGB
        # img_mat = cv.cvtColor(img_mat, cv.COLOR_BGR2RGB)

        # create new window
        finalize_new_window_image(create_window_image(label, mat, main_window))


def ev_exit(window):
    # exit program if main window has been closed
    if is_main_window(window):
        return True
    if window == get_active_window():
        set_active_window(None)
        # disable parts of menu
        set_menu_off()

    remove_from_open_windows(window)
    window.close()

    cv_destroy_window("preview")
    return False


def ev_select(window):

    set_active_window(window)
    set_menu_on()


def ev_duplicate(main_window, window):
    # get image matrix
    mat = get_mat(window)

    # get name and extension
    name, ext = os.path.splitext(window["-IMAGE-"].metadata["label"])

    # generate label
    name = name + " - Duplicate"
    label = name + ext

    # ask for label
    duplicate_label = sg.popup_get_text(
        message="Provide title for the duplicate:",
        title="Duplicate...",
        default_text=label,
        keep_on_top=True,
    )

    # create new window for duplicated image
    if duplicate_label is not None:
        finalize_new_window_image(
            (create_window_image(duplicate_label, mat, main_window))
        )


def ev_save(window, values):

    # handle save via menu
    if "-MENUBAR-" in values:
        window = get_active_window()

    # get image matrix
    img_mat = get_mat(window)

    # get image label
    label = get_label(window)

    # display popup
    filepath = sg.popup_get_file(
        message="Save file", save_as=True, no_window=True, default_path=label
    )

    if filepath:
        # save image to file
        cv.imwrite(filepath, img_mat)


def ev_histogram():

    active_window = get_active_window()

    # get image matrix
    mat = get_mat(active_window)

    # get image type
    img_type = get_type(active_window)

    # set info for initial histogram
    if img_type == "GRAYSCALE":
        calc_type = "GRAYSCALE"
        hist_params = {
            "x-axis_label": "Intensity",
            "bar_color": "black",
            "highlight_bar_color": "orange",
        }
    else:
        calc_type = "RGB_INTENSITY_WEIGHTED"
        hist_params = {
            "x-axis_label": "Intensity (weighted)",
            "bar_color": "black",
            "highlight_bar_color": "orange",
        }

    # calculate lut for histogram
    lut = calc_lut(mat, calc_type)

    # calculate lut stats for histogram
    stats = get_img_stats(lut)

    # create new histogram
    window_histogram = APO_histogram.new_histogram(img_type, lut, stats, hist_params)

    # get histogram's metadata handle
    hist_md = window_histogram.metadata

    hist_md["type"] = img_type

    # set histogram's current (active) lut and stats
    hist_md["luts"] = {
        calc_type: {"lut": lut, "stats": stats, "hist_params": hist_params}
    }

    # RGB additional precalculations
    if img_type == "RGB":

        lut_b, lut_g, lut_r = calc_lut(mat, "RGB_SEPERATE")
        hist_md["luts"]["RGB_SEPERATE_R"] = {
            "lut": lut_r,
            "stats": get_img_stats(lut_r),
            "hist_params": {
                "x-axis_label": "Red",
                "bar_color": "#fa3232",
                "highlight_bar_color": "black",
            },
        }
        hist_md["luts"]["RGB_SEPERATE_G"] = {
            "lut": lut_g,
            "stats": get_img_stats(lut_g),
            "hist_params": {
                "x-axis_label": "Green",
                "bar_color": "#32fa32",
                "highlight_bar_color": "black",
            },
        }
        hist_md["luts"]["RGB_SEPERATE_B"] = {
            "lut": lut_b,
            "stats": get_img_stats(lut_b),
            "hist_params": {
                "x-axis_label": "Blue",
                "bar_color": "#3232fa",
                "highlight_bar_color": "orange",
            },
        }


def ev_histogram_rgb(window):

    # get graph handle
    graph = window["-HISTGRAPH-"]

    # get histogram's metadata handle
    hist_md = window.metadata

    # get state of the rgb button
    rgb_state = window["-HISTRGB-"].metadata

    # get list of types of histogram
    key_list = list(hist_md["luts"].keys())

    # update state of rgb button
    rgb_state = rgb_state + 1
    if rgb_state == len(key_list):
        rgb_state = 0
    window["-HISTRGB-"].metadata = rgb_state

    # get key
    key = key_list[rgb_state]

    # set active lut
    hist_md["active_lut"] = {
        "lut": hist_md["luts"][key]["lut"],
        "stats": hist_md["luts"][key]["stats"],
        "hist_params": hist_md["luts"][key]["hist_params"],
    }

    # draw graph
    APO_histogram.draw_histogram_graph(
        graph=graph,
        img_lut=hist_md["active_lut"]["lut"],
        img_stats=hist_md["active_lut"]["stats"],
        hist_params=hist_md["active_lut"]["hist_params"],
    )

    # get stats handle
    stats = hist_md["active_lut"]["stats"]

    # update stats
    window["-HISTVAL-"].update(f"Value: ---")
    window["-HISTN-"].update(f"N: {round(stats['N'])}")
    window["-HISTMIN-"].update(f"Min: {stats['bounds'][0]}")
    window["-HISTMAX-"].update(f"Max: {stats['bounds'][1]}")
    window["-HISTCOUNT-"].update(f"Count: ---")
    window["-HISTSTDEV-"].update(f"StdDev: {round(stats['stdev'],3)}")
    window["-HISTMODE-"].update(
        f"Mode: {stats['mode_position']} ({round(stats['mode'])})"
    )
    window["-HISTMEAN-"].update(f"Mean: {round(stats['mean'],3)}")


def ev_scale(window, event):

    # matrix of the image
    mat = get_mat(window)

    # current scale state of the image
    scale_state = get_scale_state(window)

    # list of possible scales for the image
    scale_list = get_scale_list(window)

    if event == "-SCALEUP-":
        if scale_state == len(scale_list) - 1:
            return
        scale_state = scale_state + 1
    elif event == "-SCALEDOWN-":
        if scale_state == 0:
            return
        scale_state = scale_state - 1

    # check if bytes of this scale has been already calculated
    cached = get_scale_cache_img_bytes(window, scale_state)

    if cached == None:
        # not cached, calculate
        scale_percent = scale_list[scale_state]  # percent of original size

        if scale_percent != 100:
            mat = get_scaled_mat(scale_percent, mat)
            if mat is None:
                return

        # get img extension
        ext = os.path.splitext(window["-IMAGE-"].metadata["label"])[1]

        # encode matrix to bytes
        img_bytes = cv.imencode(ext, mat)[1].tobytes()

        # convert to png
        img_bytes = APO_cvt_png.cvt_bytes(img_bytes)

        set_scale_cache(window, scale_state, mat, img_bytes)
    else:
        # use cached result
        img_bytes = cached

    # display img
    window["-IMAGE-"].update(data=img_bytes)

    # update scale label
    window["-LSCALE-"].update(f"Scale: {scale_list[scale_state]}%")

    # refresh after changing content (image)
    window.refresh()

    # Update for scroll area of Column element
    window["-IMGCOLUMN-"].contents_changed()

    # update scale state
    set_scale_state(window, scale_state)


def ev_grayscale():
    apply_to_active(calc_rgb_intensity_weighted)


def ev_normalize_histogram_linear(window, event, values):

    active_window = get_active_window()

    if event == "Normalize histogram (linear)":

        mat = get_mat(active_window)
        # get intensity bounds
        mat_min = np.amin(mat)
        mat_max = np.amax(mat)

        # ask for range
        sg.Window(
            "Linear normalization output range",
            [
                [sg.Text("Provide output range for linear normalization:")],
                [
                    sg.Column(
                        [
                            [sg.Text("Input range")],
                            [
                                sg.Text("Min:", size=(5, 1)),
                                sg.Input(mat_min, key="-LINNORMMIN-", size=(3, 1)),
                            ],
                            [
                                sg.Text("Max:", size=(5, 1)),
                                sg.Input(mat_max, key="-LINNORMMAX-", size=(3, 1)),
                            ],
                        ]
                    ),
                    sg.Column(
                        [
                            [sg.Text("Output range")],
                            [
                                sg.Text("Low:", size=(5, 1)),
                                sg.Input(0, key="-LINNORMLOW-", size=(3, 1)),
                            ],
                            [
                                sg.Text("High:", size=(5, 1)),
                                sg.Input(255, key="-LINNORMHIGH-", size=(3, 1)),
                            ],
                        ]
                    ),
                ],
                [sg.Button("OK", key="-LINNORMOK-")],
            ],
            finalize=True,
            keep_on_top=True,
        )

    if event == "-LINNORMOK-":

        # close popup window
        window.close()

        # get image metadata handle
        imd = active_window["-IMAGE-"].metadata
        mat = np.array(imd["mat"])

        # get min and max input range for linear normalization
        input_range = (int(values["-LINNORMMIN-"]), int(values["-LINNORMMAX-"]))

        # get low and high of output range for linear normalization
        output_range = (int(values["-LINNORMLOW-"]), int(values["-LINNORMHIGH-"]))

        low, high = output_range

        # validate input
        if not (255 >= int(high) >= int(low) >= 0):
            sg.popup_ok(
                "Both low and high must be intigers between 0 and 255 and low cannot exceed high.",
                keep_on_top=True,
            )
            return

        # normalize
        normalized_mat = normalize_linear(mat, input_range, output_range)

        apply_transformation(active_window, normalized_mat)


def ev_normalize_histogram_nonlinear(window, event, values):

    params = {
        "prompt_event": "Normalize histogram (non-linear)",
        "event_key": "NORMHISTNONLIN",
        "val_label": "Gamma",
        "to_type": to_float,
        "default_val": 5,
        "slider_range": (0.1, 10),
        "slider_resolution": 0.1,
        "func": normalize_nonlinear,
    }

    template_slider_one(window, event, values, params)


def ev_equalize_histogram():
    apply_to_active(equalize_histogram)


def ev_negation():
    apply_to_active(negation)


def ev_thresholding_single(window, event, values):

    params = {
        "prompt_event": "Thresholding (single)",
        "event_key": "THRESHSINGLE",
        "val_label": "Intensity threshold",
        "enable_checkbox": True,
        "checkbox_label": "Zachowaj poziomy szarości",
        "func": threshold_single,
    }

    template_slider_one(window, event, values, params)


def ev_thresholding_band(window, event, values):

    params = {
        "prompt_event": "Thresholding (band)",
        "event_key": "THRESHBAND",
        "header_text": "Provide band of intensity values:",
        "enable_checkbox": True,
        "checkbox_label": "Zachowaj poziomy szarości",
        "func": threshold_band,
    }

    template_slider_two(window, event, values, params)


def ev_image_calculator(window, event, values):
    prompt_event = "Image calculator"

    if event == prompt_event:
        element_map = {}

        update_map = {
            "-IMGCALCSELECTIMGONEPREVIEW-": [],
            "-IMGCALCSELECTEDOPERATION-": [],
            "-IMGCALCSELECTIMGTWOPREVIEW-": [],
            "-IMGCALCNOSATURATIONCHECK-": [],
            "-IMGCALCPREVIEW-": [],
        }

        image_list = get_open_windows_labels_with_id()
        image_map = list(get_open_windows().keys())

        math_operations = ["add", "subtract (absolute)", "AND", "OR", "XOR"]

        prompt = sg.Window(
            "Image calculator",
            [
                [
                    sg.Text("Image 1: "),
                    sg.Combo(
                        values=image_list,
                        default_value=image_list[0],
                        # s=(15, 22),
                        enable_events=True,
                        readonly=True,
                        k="-IMGCALCSELECTIMGONEPREVIEW-",
                    ),
                ],
                [
                    sg.Text("Operation: "),
                    sg.Combo(
                        values=math_operations,
                        default_value=math_operations[0],
                        s=(15, 22),
                        enable_events=True,
                        readonly=True,
                        k="-IMGCALCSELECTEDOPERATION-",
                    ),
                ],
                [
                    sg.Text("Image 2: "),
                    sg.Combo(
                        values=image_list,
                        default_value=image_list[0],
                        # s=(15, 22),
                        enable_events=True,
                        readonly=True,
                        k="-IMGCALCSELECTIMGTWOPREVIEW-",
                    ),
                ],
                [sg.Text("✅ Images are of equal size.", key="-IMGCALCWARNING-")],
                [
                    sg.Checkbox(
                        "no saturation",
                        key="-IMGCALCNOSATURATIONCHECK-",
                        enable_events=True,
                        pad=(0, 10),
                    )
                ],
                [sg.Button("Preview", key="-IMGCALCPREVIEW-")],
                [sg.Button("OK", key="-IMGCALCOKAPL-")],
            ],
            modal=True,
            keep_on_top=True,
            size=(300, 220),
            metadata={
                "prompt_event": prompt_event,
                "element_map": element_map,
                "update_map": update_map,
                "function": img_calc,
                "image_map": image_map,
            },
        )
        transform(prompt, event, values)
    else:
        values["-IMGCALCSELECTIMGONEPREVIEW-"] = get_mat_by_id(
            window.metadata["image_map"][
                window["-IMGCALCSELECTIMGONEPREVIEW-"].widget.current()
            ]
        )
        values["-IMGCALCSELECTIMGTWOPREVIEW-"] = get_mat_by_id(
            window.metadata["image_map"][
                window["-IMGCALCSELECTIMGTWOPREVIEW-"].widget.current()
            ]
        )

        values["-IMGCALCEVENTKEY-"] = event
        values["-IMGCALCWINDOW-"] = window

        
        set_active_window(
            get_window_by_id(
                window.metadata["image_map"][
                    window["-IMGCALCSELECTIMGONEPREVIEW-"].widget.current()
                ]
            )
        )
            # set_active_window(get_window_by_id(window.metadata["image_map"][0]))

        transform(window, event, values)

def ev_not():
    apply_to_active(APO_negation_NOT)

def ev_add(window, event, values):

    params = {
        "prompt_event": "Add",
        "event_key": "MATHADD",
        "enable_checkbox": True,
        "checkbox_label": "no saturation",
        "func": math_add,
    }

    template_slider_one(window, event, values, params)


def ev_multiply(window, event, values):

    params = {
        "prompt_event": "Multiply",
        "event_key": "MATHMULTIPLY",
        "to_type": to_float,
        "default_val": 5,
        "slider_range": (0.1, 10),
        "slider_resolution": 0.1,
        "enable_checkbox": True,
        "checkbox_label": "no saturation",
        "func": math_multiply,
    }

    template_slider_one(window, event, values, params)


def ev_divide(window, event, values):

    params = {
        "prompt_event": "Divide",
        "event_key": "MATHDIVIDE",
        "to_type": to_float,
        "default_val": 5,
        "slider_range": (0.1, 10),
        "slider_resolution": 0.1,
        "enable_checkbox": True,
        "checkbox_label": "no saturation",
        "func": math_divide,
    }

    template_slider_one(window, event, values, params)


def ev_thresholding_adaptive():
    apply_to_active(threshold_adaptive)


def ev_thresholding_otsu():
    apply_to_active(threshold_otsu)


def ev_line_detection_canny(window, event, values):

    params = {
        "prompt_event": "Line detection (Canny)",
        "event_key": "LINEDETCANNY",
        "enable_checkbox": False,
        "func": line_detection_canny,
    }

    template_slider_two(window, event, values, params)


def ev_line_detection_prewitt():
    apply_to_active(line_detection_prewitt)


def ev_line_detection_sobel():
    apply_to_active(line_detection_sobel)


def ev_morphology_dilation():
    apply_to_active(morphology_dilation)


def ev_morphology_erosion():
    apply_to_active(morphology_erosion)


def ev_morphology_opening():
    apply_to_active(morphology_opening)


def ev_morphology_closing():
    apply_to_active(morphology_closing)
