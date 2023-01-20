import PySimpleGUI as sg

import init
import APO_events

prev_lit = None

while True:

    window, event, values = sg.read_all_windows()
    if (
        True
        # event != None
        # and event not in ("-HISTGRAPH-+MOVE", "-LONGOPERATION-")
        # and "SLIDER" not in event
    ):
        print(
            30 * "#"
            + f"\nwindow: {(str(window)).split('at',1)[1]}"
            + f"\nevent: {event}"
            + f"\nvalues{values}"
        )

    if event == "Exit" or event == sg.WIN_CLOSED:
        if APO_events.ev_exit(window):
            break

    elif event == "Select":
        APO_events.ev_select(window)

    elif event == "Open":
        APO_events.ev_open(init.main_window)

    elif event == "Duplicate":
        APO_events.ev_duplicate(init.main_window, window)

    elif event == "Save":
        APO_events.ev_save(window, values)

    elif event == "-HISTGRAPH-+MOVE":
        # CLEAN UP THIS CODE

        # get graph handle
        graph = window["-HISTGRAPH-"]
        # number of figures drawn on the histogram graph
        FIGURES_NUM = 7
        # unlit previously lit bar
        if prev_lit != None:
            graph.send_figure_to_back(prev_lit)

        # get mouse x,y coordinates
        x, y = values["-HISTGRAPH-"]

        adjustment = graph.metadata["last_id"]
        figures = graph.get_figures_at_location((x, y))
        if len(figures) != 0 and figures[0] >= FIGURES_NUM:
            # get focused bar
            if len(figures) == 1:
                figure = figures[0]
            elif len(figures) > 1:
                figure = max(figures)

            # highlight focused bar (send normal bar to back)
            graph.send_figure_to_back(figure - 256)

            # set prev to highlight bar
            prev_lit = figure - 256 - 256

            # obtain count and value
            img_lut = window.metadata["active_lut"]["lut"]
            val = figure - 256 - 256 - FIGURES_NUM - adjustment

            if val not in range(0, 256):
                continue
            count = int(img_lut[val])

            # print count and value
            window["-HISTVAL-"].update(f"Value: {val}")
            window["-HISTCOUNT-"].update(f"Count: {count}")
        continue

    elif event == "Histogram":
        APO_events.ev_histogram()

    elif event == "-HISTRGB-":
        APO_events.ev_histogram_rgb(window)

    elif event in ("-SCALEDOWN-", "-SCALEUP-"):
        APO_events.ev_scale(window, event)

    elif event in ("Normalize histogram (linear)", "-LINNORMOK-"):
        APO_events.ev_normalize_histogram_linear(window, event, values)

    elif event == "Grayscale":
        APO_events.ev_grayscale()

    elif event in ("Normalize histogram (non-linear)") or event.startswith("-NORMHISTNONLIN"):
        APO_events.ev_normalize_histogram_nonlinear(window, event, values)

    elif event in ("Equalize histogram"):
        APO_events.ev_equalize_histogram()

    elif event in ("Negation"):
        APO_events.ev_negation()

    elif event in ("Thresholding (single)") or event.startswith("-THRESHSINGLE"):
        APO_events.ev_thresholding_single(window, event, values)

    elif event in ("Thresholding (band)") or event.startswith("-THRESHBAND"):
        APO_events.ev_thresholding_band(window, event, values)

    elif event in ("Image calculator") or event.startswith("-IMGCALC"):
        APO_events.ev_image_calculator(window, event, values)

    elif event in ("NOT"):
        APO_events.ev_not()

    elif event in ("Add") or event.startswith("-MATHADD"):
        APO_events.ev_add(window, event, values)
        
    elif event in ("Multiply") or event.startswith("-MATHMULTIPLY"):
        APO_events.ev_multiply(window, event, values)
        
    elif event in ("Divide") or event.startswith("-MATHDIVIDE"):
        APO_events.ev_divide(window, event, values)

    elif event in ("Thresholding (adaptive)"):
        APO_events.ev_thresholding_adaptive()

    elif event in ("Thresholding (Otsu)"):
        APO_events.ev_thresholding_otsu()

    elif event in ("Line detection (Canny)") or event.startswith("-LINEDETCANNY"):
        APO_events.ev_line_detection_canny(window, event, values)

    elif event in ("Line detection (Prewitt)"):
        APO_events.ev_line_detection_prewitt()

    elif event in ("Line detection (Sobel)"):
        APO_events.ev_line_detection_sobel()

    elif event in ("Dilation"):
        APO_events.ev_morphology_dilation()

    elif event in ("Erosion"):
        APO_events.ev_morphology_erosion()

    elif event in ("Opening"):
        APO_events.ev_morphology_opening()

    elif event in ("Closing"):
        APO_events.ev_morphology_closing()











    # used for histogram highlight (reset id of previously lit bar)
    prev_lit = None

init.main_window.close()
