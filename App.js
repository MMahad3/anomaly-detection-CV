import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import "./App.css"; 
import demoVideo from "./assets/videos/ANOMALY DETECTION.mp4"; 
import logo from "./assets/images/logo.jpg"; 
import mahad from "./assets/images/mahad.png";
import asad from "./assets/images/asad.png";
import taha from "./assets/images/taha.png";

function Front() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [classificationResult, setClassificationResult] = useState("");
  const [showSplash, setShowSplash] = useState(true); // State to control splash screen visibility

  // Simulate splash screen duration
  useEffect(() => {
    const splashTimeout = setTimeout(() => setShowSplash(false), 8000); // 8 seconds
    return () => clearTimeout(splashTimeout); // Cleanup timeout
  }, []);

  // Handle file upload (video or image) and send it to the backend
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://127.0.0.1:5000/classify", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Error in classification");
        }

        const data = await response.json();
        setClassificationResult(data.result || "No result received");
        alert(`Detected activity: ${data.result}`);
      } catch (error) {
        console.error("Error during classification:", error);
        setClassificationResult("Error during classification");
      }
    }
  };

  const triggerFileUpload = () => {
    document.getElementById("fileUploadInput").click();
  };

  // Splash Screen
if (showSplash) {
  return (
    <div className="splash-screen">
      {/* Animated Logo */}
      <motion.div
        initial={{ scale: 0, opacity: 0, y: -200 }}
        animate={{ scale: 1.5, opacity: 1, y: 0 }}
        transition={{
          ease: "easeOut",
          duration: 1.5,
          type: "spring",
        }}
        className="splash-logo-container"
      >
        <img src={logo} alt="Logo" className="splash-logo" />
      </motion.div>

      {/* Animated Text Message */}
      <div className="splash-message-container">
        {"Your Gateway Towards Safety".split(" ").map((word, index) => (
          <motion.span
            key={index}
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              ease: "easeOut",
              duration: 0.8,
              delay: 1 + index * 0.3,
            }}
            className={`splash-message-word`}
          >
            {word}
          </motion.span>
        ))}
      </div>
    </div>
  );
}


 

  return (
    <div className="App">
      {/* Header */}
      <header className="App-header">
        <div className="logo-container">
          <img src={logo} alt="Guardian Vision Logo" className="logo" />
        </div>
        <h1>GUARDIAN VISION: IMPROVING PUBLIC SAFETY</h1>
        <nav>
          <a href="#about">About</a>
          <a href="#features">Features</a>
          <a href="#how-it-works">How It Works</a>
        </nav>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Video and Introduction */}
        <section className="video-intro-section">
          <div className="video-container">
            <video autoPlay loop muted className="video">
              <source src={demoVideo} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>

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
              <button className="test-file-button" onClick={triggerFileUpload}>
                Test Video or Image
              </button>
              <input
                type="file"
                id="fileUploadInput"
                accept="image/*,video/mp4,video/x-m4v,video/"
                style={{ display: "none" }}
                onChange={handleFileUpload}
              />
            </div>
            <div className="result-section">
              <p>{classificationResult || "Upload a video to see the result"}</p>
            </div>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="features-section">
          <h1>Features</h1>
          <p>
            Our anomaly detection platform is built to enhance public safety
            through real-time, AI-powered video analysis. Here’s how it stands
            out:
          </p>
          <ul>
            <li>Real-Time Detection</li>
            <li>Context-Aware Analysis</li>
            <li>Human Behavior Recognition</li>
            <li>Scalable & Adaptive</li>
            <li>Efficient Alerts</li>
            <li>User-Friendly Interface</li>
            <li>Continuous Improvement</li>
            <li>Data Security</li>
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


        {/* About Us */}
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
      </main>
    </div>
  );
}

export default Front;
