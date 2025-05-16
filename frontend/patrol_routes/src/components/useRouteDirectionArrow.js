// This custom hook adds arrow decorators to show route direction
import { useEffect } from "react";
import L from "leaflet";
import "leaflet-polylinedecorator";

const useRouteDirectionArrows = (map, routeInfo, colors) => {
  useEffect(() => {
    if (!map || !routeInfo || !routeInfo.routes) return;

    // Remove any existing decorators
    map.eachLayer((layer) => {
      if (layer._arrowDecoration) {
        map.removeLayer(layer);
      }
    });

    // Add new decorators for each route
    routeInfo.routes.forEach((route, i) => {
      if (route.geometry && route.geometry.length > 1) {
        const polyline = L.polyline(
          route.geometry.map((pos) => [pos[1], pos[0]]),
          { opacity: 0 } // Make the base polyline invisible
        );

        const arrowHead = L.Symbol.arrowHead({
          pixelSize: 12,
          pathOptions: {
            fillOpacity: 1,
            weight: 2,
            color: colors[i % colors.length],
            fillColor: colors[i % colors.length],
          },
        });

        const decorator = L.polylineDecorator(polyline, {
          patterns: [
            {
              offset: "10%",
              repeat: "20%",
              symbol: arrowHead,
            },
          ],
        }).addTo(map);

        // Add a property to identify this layer for future cleanup
        decorator._arrowDecoration = true;
      }
    });

    // Cleanup function
    return () => {
      map.eachLayer((layer) => {
        if (layer._arrowDecoration) {
          map.removeLayer(layer);
        }
      });
    };
  }, [map, routeInfo]);
};

export default useRouteDirectionArrows;
