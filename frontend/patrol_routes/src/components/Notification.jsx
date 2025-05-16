import { useState, useEffect } from "react";

function Notification({ message, type, onClose, duration = 3000, style = {} }) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(false);
      setTimeout(() => {
        onClose();
      }, 300); // Allow fade-out animation to complete
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const getBackgroundColor = () => {
    switch (type) {
      case "success":
        return "#4a7d1a";
      case "error":
        return "#7d1a1a";
      case "info":
        return "#1a4a7d";
      case "warning":
        return "#7d4a1a";
      default:
        return "#4a1a1a";
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        top: "20px",
        right: "20px",
        padding: "12px 10px",
        backgroundColor: getBackgroundColor(),
        color: "white",
        borderRadius: "4px",
        boxShadow: "0 2px 10px rgba(0, 0, 0, 0.3)",
        zIndex: 1000,
        opacity: visible ? 1 : 0,
        transition: "opacity 0.3s ease",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        maxWidth: "350px",
        ...style,
      }}
    >
      <span>{message}</span>
      <button
        onClick={() => {
          setVisible(false);
          setTimeout(onClose, 200);
        }}
        style={{
          background: "none",
          border: "none",
          color: "white",
          fontSize: "16px",
          cursor: "pointer",
          marginLeft: "5px",
        }}
      >
        ×
      </button>
    </div>
  );
}

export default Notification;
