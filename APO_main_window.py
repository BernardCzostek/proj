import PySimpleGUI as sg

def create_main_window():

    # set GUI theme
    sg.theme("Default1")

    # define layout
    menu = [sg.Menu([], k="-MENUBAR-")]

    # define layout
    layout = [menu, [sg.Text("idk", key="main_text")]]

    main_window = sg.Window(
        "Window",
        layout,
        keep_on_top=True,
        size=(300, 70),
        finalize=True,
        metadata={"main_window": True, 'active_window': None, "window_count": 0, "open_windows": {}},
        relative_location=(-300, 0),
    )

    # since no image has been selected yet set menu to off
    from APO_functions import set_menu_off
    set_menu_off(main_window)

    return main_window