# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import dearpygui.dearpygui as dpg


class App:
    """
    Base class for creating the GUI with DearPyGUI.

    This class provides the basic structure for creating a window, setting up
    the application, and handling the main loop. Derived classes should implement
    the create_app_body and create_network_iface methods.
    """

    def __init__(self):
        """Initialize the application with default window settings."""
        self.window_name = "DearPyGUI App"
        self.window_width = 800
        self.window_height = 600
        self.resizeable = False

    def create_app_body(self):
        """
        Create the body of the application.

        This method should be implemented by derived classes to define the
        UI elements of the application.

        Raises:
            NotImplementedError: If the derived class does not implement this method.
        """
        raise NotImplementedError

    def create_network_iface(self):
        """
        Create the network interface for the application.

        This method should be implemented by derived classes to set up
        network communication.

        Raises:
            NotImplementedError: If the derived class does not implement this method.
        """
        raise NotImplementedError

    def _create_app(self) -> None:
        """
        Create the application window.
        """
        # Initialize DearPyGUI
        dpg.create_context()
        dpg.create_viewport(
            title=self.window_name,
            width=self.window_width,
            height=self.window_height,
            resizable=self.resizeable,
        )
        dpg.setup_dearpygui()

        # Set up fonts
        self.font_scale = 20
        with dpg.font_registry():
            font_medium = dpg.add_font("./isaac_zmq_server/fonts/Inter-Medium.ttf", 16 * self.font_scale)

        dpg.set_global_font_scale(1 / self.font_scale)
        dpg.bind_font(font_medium)

        # Create the application body
        self.create_app_body()

        # Show the viewport
        dpg.show_viewport()

    def _run(self) -> None:
        """
        Run the main application loop.
        """
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

    def _cleanup(self) -> None:
        """
        Clean up resources when the application is closed.

        This method cleans up the ZMQ server and destroys the DearPyGUI context.
        """
        self.zmq_server.cleanup()
        dpg.destroy_context()

    @classmethod
    def run_app(cls) -> None:
        """
        Static method to run the application.
        """
        app = cls()
        app._create_app()
        app.create_network_iface()
        app._run()
        app._cleanup()
