from streamlit.elements.map import (
    DeltaGenerator,
    DeckGlJsonChartProto,
    to_deckgl_json,
    Data,
    Color,
    marshall,
    MapMixin,
)


class CustomMapMixin(MapMixin):

    def map(
        self,
        data: Data = None,
        *,
        latitude: str | None = None,
        longitude: str | None = None,
        color: None | str | Color = None,
        size: None | str | float = None,
        zoom: int | None = None,
        use_container_width: bool = True,
        width: int | None = None,
        height: int | None = None,
        map_style: str | None = None,
    ) -> DeltaGenerator:
        """Display a map with a scatterplot overlaid onto it.

        This is a wrapper around ``st.pydeck_chart`` to quickly create
        scatterplot charts on top of a map, with auto-centering and auto-zoom.

        When using this command, Mapbox provides the map tiles to render map
        content. Note that Mapbox is a third-party product and Streamlit accepts
        no responsibility or liability of any kind for Mapbox or for any content
        or information made available by Mapbox.

        Mapbox requires users to register and provide a token before users can
        request map tiles. Currently, Streamlit provides this token for you, but
        this could change at any time. We strongly recommend all users create and
        use their own personal Mapbox token to avoid any disruptions to their
        experience. You can do this with the ``mapbox.token`` config option. The
        use of Mapbox is governed by Mapbox's Terms of Use.

        To get a token for yourself, create an account at https://mapbox.com.
        For more info on how to set config options, see
        https://docs.streamlit.io/develop/api-reference/configuration/config.toml.

        Parameters
        ----------
        data : Anything supported by st.dataframe
            The data to be plotted.

        latitude : str or None
            The name of the column containing the latitude coordinates of
            the datapoints in the chart.

            If None, the latitude data will come from any column named 'lat',
            'latitude', 'LAT', or 'LATITUDE'.

        longitude : str or None
            The name of the column containing the longitude coordinates of
            the datapoints in the chart.

            If None, the longitude data will come from any column named 'lon',
            'longitude', 'LON', or 'LONGITUDE'.

        color : str or tuple or None
            The color of the circles representing each datapoint.

            Can be:

            - None, to use the default color.
            - A hex string like "#ffaa00" or "#ffaa0088".
            - An RGB or RGBA tuple with the red, green, blue, and alpha
            components specified as ints from 0 to 255 or floats from 0.0 to
            1.0.
            - The name of the column to use for the color. Cells in this column
            should contain colors represented as a hex string or color tuple,
            as described above.

        size : str or float or None
            The size of the circles representing each point, in meters.

            This can be:

            - None, to use the default size.
            - A number like 100, to specify a single size to use for all
            datapoints.
            - The name of the column to use for the size. This allows each
            datapoint to be represented by a circle of a different size.

        zoom : int
            Zoom level as specified in
            https://wiki.openstreetmap.org/wiki/Zoom_levels.

        use_container_width : bool
            Whether to override the map's native width with the width of
            the parent container. If ``use_container_width`` is ``True``
            (default), Streamlit sets the width of the map to match the width
            of the parent container. If ``use_container_width`` is ``False``,
            Streamlit sets the width of the chart to fit its contents according
            to the plotting library, up to the width of the parent container.

        width : int or None
            Desired width of the chart expressed in pixels. If ``width`` is
            ``None`` (default), Streamlit sets the width of the chart to fit
            its contents according to the plotting library, up to the width of
            the parent container. If ``width`` is greater than the width of the
            parent container, Streamlit sets the chart width to match the width
            of the parent container.

            To use ``width``, you must set ``use_container_width=False``.

        height : int or None
            Desired height of the chart expressed in pixels. If ``height`` is
            ``None`` (default), Streamlit sets the height of the chart to fit
            its contents according to the plotting library.

        Examples
        --------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> df = pd.DataFrame(
        ...     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        ...     columns=["lat", "lon"],
        ... )
        >>> st.map(df)

        .. output::
        https://doc-map.streamlit.app/
        height: 600px

        You can also customize the size and color of the datapoints:

        >>> st.map(df, size=20, color="#0044ff")

        And finally, you can choose different columns to use for the latitude
        and longitude components, as well as set size and color of each
        datapoint dynamically based on other columns:

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": np.random.randn(1000) / 50 + 37.76,
        ...         "col2": np.random.randn(1000) / 50 + -122.4,
        ...         "col3": np.random.randn(1000) * 100,
        ...         "col4": np.random.rand(1000, 4).tolist(),
        ...     }
        ... )
        >>>
        >>> st.map(df, latitude="col1", longitude="col2", size="col3", color="col4")

        .. output::
        https://doc-map-color.streamlit.app/
        height: 600px

        """
        # This feature was turned off while we investigate why different
        # map styles cause DeckGL to crash.
        #
        # For reference, this was the docstring for map_style:
        #
        #   map_style : str or None
        #       One of Mapbox's map style URLs. A full list can be found here:
        #       https://docs.mapbox.com/api/maps/styles/#mapbox-styles
        #
        #       This feature requires a Mapbox token. See the top of these docs
        #       for information on how to get one and set it up in Streamlit.
        #
        map_style = map_style
        map_proto = DeckGlJsonChartProto()
        deck_gl_json = to_deckgl_json(
            data, latitude, longitude, size, color, map_style, zoom
        )
        marshall(
            map_proto, deck_gl_json, use_container_width, width=width, height=height
        )
        return self.dg._enqueue("deck_gl_json_chart", map_proto)
