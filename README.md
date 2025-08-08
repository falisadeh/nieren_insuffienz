# nieren_insuffienz
import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';

// Das gesamte App-Layout wird in dieser Komponente definiert.
const App = () => {
  // Zustand für die Anzeige des Overlays
  const [showOverlay, setShowOverlay] = useState(false);
  // Zustand für die rSO2-Daten, die in der Grafik dargestellt werden
  const [rso2Data, setRso2Data] = useState([]);
  // Referenz auf das Canvas-Element für die Grafik
  const canvasRef = useRef(null);
  // Zustand für den aktuellsten rSO2-Wert
  const [currentRso2, setCurrentRso2] = useState(75);

  // useEffect-Hook, um die Grafikdaten zu simulieren und zu aktualisieren.
  useEffect(() => {
    // Initialisiere die Daten mit einigen Werten
    setRso2Data(Array.from({ length: 20 }, () => 70 + Math.random() * 20 - 10));

    // Simuliere alle 3 Sekunden neue Daten
    const interval = setInterval(() => {
      setRso2Data(prevData => {
        // Entferne den ältesten Wert und füge einen neuen hinzu
        const newData = [...prevData.slice(1), 70 + Math.random() * 20 - 10];
        // Aktualisiere den aktuellen Wert
        setCurrentRso2(newData[newData.length - 1].toFixed(1));
        return newData;
      });
    }, 3000);

    // Bereinigungsfunktion, um das Intervall zu löschen, wenn die Komponente unmontiert wird
    return () => clearInterval(interval);
  }, []);

  // useEffect-Hook, um die Grafik zu zeichnen, wenn die Daten oder das Overlay sich ändern
  useEffect(() => {
    if (showOverlay && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      const drawGraph = () => {
        // Lösche das Canvas vor dem Neuzeichnen
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Definiere die Grafik-Eigenschaften
        const padding = 10;
        const graphWidth = canvas.width - 2 * padding;
        const graphHeight = canvas.height - 2 * padding;
        
        // Finde den Min- und Max-Wert für die Skalierung
        const maxVal = Math.max(...rso2Data);
        const minVal = Math.min(...rso2Data);
        const valueRange = maxVal - minVal;

        // Zeichne die Linie
        ctx.beginPath();
        ctx.strokeStyle = '#38bdf8'; // Helle blaue Linie
        ctx.lineWidth = 2;
        
        rso2Data.forEach((value, index) => {
          // Berechne die x- und y-Koordinaten
          const x = padding + (index / (rso2Data.length - 1)) * graphWidth;
          const y = padding + graphHeight - ((value - minVal) / valueRange) * graphHeight;

          if (index === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();

        // Fülle den Bereich unter der Linie
        ctx.lineTo(canvas.width - padding, canvas.height - padding);
        ctx.lineTo(padding, canvas.height - padding);
        ctx.closePath();
        ctx.fillStyle = 'rgba(56, 189, 248, 0.2)'; // Halbtransparente Füllung
        ctx.fill();
      };
      drawGraph();
    }
  }, [rso2Data, showOverlay]);

  return (
    <div className="bg-gray-900 min-h-screen flex items-center justify-center p-4">
      {/* Container für das Haupt-Widget */}
      <div className="relative w-full max-w-sm bg-gray-800 rounded-2xl shadow-xl p-6 flex flex-col items-center space-y-4 text-white">
        <h1 className="text-2xl font-bold mb-4">Medizinische Daten-Visualisierung</h1>

        {/* Nieren-Visualisierung */}
        <div className="relative flex items-center justify-center w-48 h-48">
          {/* Nieren-SVG (Nierenform) */}
          <svg
            className="w-full h-full text-blue-500 fill-current opacity-70"
            viewBox="0 0 100 100"
          >
            <path d="M 50,5 C 25,10 5,40 5,60 C 5,80 20,95 50,95 C 80,95 95,80 95,60 C 95,40 75,10 50,5 Z" />
          </svg>

          {/* Ein-/Ausschaltknopf für das Overlay */}
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className="absolute bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full shadow-lg transition-all"
          >
            {showOverlay ? 'Overlay ausblenden' : 'Overlay anzeigen'}
          </button>

          {/* Overlay, das nur angezeigt wird, wenn showOverlay true ist */}
          {showOverlay && (
            <div className="absolute inset-0 bg-gray-900/90 rounded-2xl p-4 flex flex-col items-center justify-center transition-opacity duration-300">
              <h2 className="text-lg font-semibold text-sky-400 mb-2">
                rSO₂ (regionale Sauerstoffsättigung)
              </h2>
              <div className="text-4xl font-extrabold text-white mb-4">
                {currentRso2}%
              </div>
              
              {/* Canvas für die dynamische Grafik */}
              <div className="w-full h-24 bg-gray-700 rounded-lg p-2">
                <canvas ref={canvasRef} className="w-full h-full" />
              </div>
              
              <p className="text-sm text-gray-400 mt-2 text-center">
                Simulierte Daten der letzten 60 Sekunden.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Startpunkt für die React-Anwendung
const root = createRoot(document.getElementById('root'));
root.render(<App />);

// Stelle sicher, dass die Tailwind-Skripte geladen sind und die root-div vorhanden ist
// Die folgende HTML wird automatisch vom Canvas-Renderer bereitgestellt
/*
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: 'Inter', sans-serif; }
  </style>
</head>
<body>
  <div id="root"></div>
</body>
</html>
*/
