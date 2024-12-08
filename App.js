import React, { useState } from "react";
import "./App.css";
import demoVideo from "./assets/videos/demo-video.mp4"; // Replace with your actual video path
import logo from "./assets/images/logo.jpg"; // Replace with your logo image

function App() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [result, setResult] = useState("");

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedVideo(file);
      const formData = new FormData();
      formData.append("video", file);

      try {
        const response = await fetch("http://127.0.0.1:5000/classify", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        setResult(data.result);
      } catch (error) {
        console.error("Error uploading video:", error);
        setResult("Error processing the video");
      }
    }
  };

  const handleTestVideo = () => {
    document.getElementById("videoUploadInput").click();
  };

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} alt="Logo" className="App-logo" />
        <h1>Guardian Vision: Improving Public Safety</h1>
      </header>
      <main>
        <section className="intro-section">
          <video autoPlay loop muted className="intro-video">
            <source src={demoVideo} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
          <button className="upload-button" onClick={handleTestVideo}>
            Upload Video
          </button>
          <input
            type="file"
            id="videoUploadInput"
            accept="video/*"
            style={{ display: "none" }}
            onChange={handleVideoUpload}
          />
          <p>{result && `Classification Result: ${result}`}</p>
        </section>
      </main>
      <footer>
        <p>&copy; 2024 Guardian Vision. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
