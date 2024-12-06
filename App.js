import React, { useState } from "react";
import "./App.css"; // Import the CSS file
import demoVideo from "./assets/videos/ANOMALY DETECTION.mp4"; // Import your video file
import logo from "./assets/images/logo.jpg"; // Import your logo image
import mahad from "./assets/images/mahad.png";
import asad from "./assets/images/asad.png";
import taha from "./assets/images/taha.png";

function App() {
  const [selectedVideo, setSelectedVideo] = useState(null);

  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedVideo(file);
      console.log("Video ready for upload:", file.name); // This will be replaced by your backend API call later
    }
  };

  const handleTestVideo = () => {
    document.getElementById("videoUploadInput").click();
  };

  return (
    <div className="App">
      {/* Header with navigation */}
      <header className="App-header">
        <div className="logo-container">
          <img src={logo} alt="Company Logo" className="logo" />
        </div>
        <h1>GUARDIAN VISION : IMPROVING PUBLIC SAFETY</h1>
        <nav>
          <a href="#about">About</a>
          <a href="#features">Features</a>
          <a href="#how-it-works">How It Works</a>
        </nav>
      </header>

      {/* Main content area */}
      <div className="main-content">
        {/* Video and Introduction Section */}
        <section className="video-intro-section">
          {/* Left Column: Video */}
          <div className="video-container">
            <video autoPlay loop muted className="video">
              <source src={demoVideo} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>

          {/* Right Column: Introduction */}
          <div className="intro-container">
            <h2>Introduction</h2>
            <p>
              Welcome to our anomaly detection platform. Our system leverages
              advanced AI-driven technology for real-time human behavior anomaly
              detection. By incorporating context-based analysis, it accurately
              identifies unusual patterns and activities in video data. Designed
              to improve public safety, it empowers organizations to make
              informed decisions and create safer environments for communities.
            </p>
            <div className="test-button">
              <button className="test-video-button" onClick={handleTestVideo}>
                Test Video
              </button>
              <input
                type="file"
                id="videoUploadInput"
                accept="video/mp4,video/x-m4v,video/*"
                style={{ display: "none" }}
                onChange={handleVideoUpload}
              />
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="features-section">
          <h1>Features</h1>
          <p>
            Our anomaly detection platform is built to enhance public safety
            through real-time, AI-powered video analysis. Here’s how it stands
            out:
          </p>
          <ul>
            <li>
              <h4>Real-Time Detection</h4>
              Instantly detects unusual behavior or suspicious activities,
              ensuring immediate action.
            </li>
            <li>
              <h4>Context-Aware Analysis</h4>
              Understands the context of each scenario to reduce false alarms
              and improve accuracy.
            </li>
            <li>
              <h4>Human Behavior Recognition</h4>
              Identifies abnormal actions like aggression or distress through
              advanced AI models.
            </li>
            <li>
              <h4>Scalable & Adaptive</h4>
              Easily integrates with existing systems and adapts to various
              environments.
            </li>
            <li>
              <h4>Efficient Alerts</h4>
              Provides clear, actionable alerts, allowing for quick response and
              informed decisions.
            </li>
            <li>
              <h4>User-Friendly Interface</h4>
              Simplifies monitoring and management with an intuitive,
              easy-to-navigate interface.
            </li>
            <li>
              <h4>Continuous Improvement</h4>
              Our system learns from every detected anomaly, improving its
              accuracy over time.
            </li>
            <li>
              <h4>Data Security</h4>
              We ensure top-tier encryption and privacy, keeping your data safe
              at all times.
            </li>
          </ul>
        </section>

        {/* How it Works Section */}
        <section id="how-it-works" className="how-it-works-section">
          <h2>How It Works</h2>
          <p>
            Our anomaly detection system leverages cutting-edge AI and computer
            vision technologies to analyze video data in real-time. Here’s an
            overview of the process:
          </p>
          <ol>
            <li>
              <strong>Data Collection:</strong>
              <p>
                The system receives continuous video input from surveillance
                cameras or video feeds, focusing on human activity in various
                environments.
              </p>
            </li>
            <li>
              <strong>Preprocessing:</strong>
              <p>
                <strong>Frame Extraction:</strong> Video frames are extracted at
                regular intervals for detailed analysis.
                <br />
                <strong>Noise Reduction:</strong> Image preprocessing
                techniques filter out irrelevant details, enhancing the focus on
                human movements.
              </p>
            </li>
            <li>
              <strong>Human Behavior Detection:</strong>
              <p>
                Using AI-powered models (e.g., Hybrid Vision Transformers), the
                system detects human activities such as walking, running, or
                sitting.
                <br />
                It leverages advanced pose detection algorithms to track and
                analyze human body movements in the scene.
              </p>
            </li>
            <li>
              <strong>Contextual Analysis:</strong>
              <p>
                The system goes beyond simple motion detection by analyzing
                context—such as the environment, time of day, and location—to
                identify unusual behavior. For example, a person running in an
                area after hours may trigger an alert.
              </p>
            </li>
            <li>
              <strong>Anomaly Identification:</strong>
              <p>
                The system compares real-time behavior with established
                patterns, detecting anomalies such as aggression, violence, or
                distress.
                <br />
                Machine learning models constantly refine detection accuracy by
                learning from previous data, ensuring fewer false alarms.
              </p>
            </li>
            <li>
              <strong>Alert System:</strong>
              <p>
                When an anomaly is detected, the system generates real-time
                alerts, notifying authorized personnel or triggering pre-set
                actions (e.g., recording or sending an alert).
              </p>
            </li>
            <li>
              <strong>Continuous Improvement:</strong>
              <p>
                The system uses feedback from detected anomalies to train and
                refine its models, ensuring better detection and fewer false
                positives with each use.
              </p>
            </li>
            <li>
              <strong>Data Security:</strong>
              <p>
                All video and user data is encrypted and securely stored,
                ensuring compliance with privacy standards and protecting
                sensitive information.
              </p>
            </li>
          </ol>
        </section>

        {/* About Section */}
        <section id="about" className="about-section">
          <h2>About Us</h2>
          <p>
            We are a team of final-year Computer Science students from FAST
            University, united by our commitment to innovation and public
            safety. Our project, "Contributing in Public Safety: A Computer
            Vision Approach for Anomaly Detection," focuses on utilizing
            advanced AI technologies to address critical challenges in human
            behavior anomaly detection.
          </p>

          <div className="team-members">
            <div className="team-member">
              <img src={mahad} alt="Mahad Munir" className="team-photo" />
              <p>Mahad Munir (21k-3388)</p>
            </div>

            <div className="team-member">
              <img src={taha} alt="Taha Ahmad" className="team-photo" />
              <p>Taha Ahmad (21k-4833)</p>
            </div>

            <div className="team-member">
              <img src={asad} alt="Asad Noor" className="team-photo" />
              <p>Asad Noor (21k-4678)</p>
            </div>
          </div>

          <p>
            With this initiative, we aim to contribute to creating safer
            communities through the power of AI and computer vision.
          </p>
        </section>

        {/* Footer */}
        <footer className="App-footer">
          <p>&copy; 2024 Your Company. All rights reserved.</p>
          <a href="#about">Privacy Policy</a>
        </footer>
      </div>
    </div>
  );
}

export default App;
