import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import time
from collections import deque
import math
import torch

# Import required packages
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    st.error(f"Missing YOLO dependency: {e}")
    YOLO_AVAILABLE = False

# Fix for PyTorch 2.6+ security restrictions
try:
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
except Exception as e:
    print(f"Safe globals warning: {e}")

# Configuration
class ModelConfig:
    MODEL_NAME = "yolov8n.pt"
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    TEMPORAL_WINDOW = 10
    MOVEMENT_THRESHOLD = 15.0
    PERSON_CLASS_ID = 0

class AppConfig:
    STREAM_WIDTH = 800
    STREAM_HEIGHT = 600
    WARNING_COOLDOWN = 5

class Colors:
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)

# Temporal Analysis Classes
class SimplePatientTracker:
    def __init__(self, patient_id, initial_bbox, temporal_window=10):
        self.patient_id = patient_id
        self.temporal_window = temporal_window
        self.position_history = deque(maxlen=temporal_window)
        self.movement_history = deque(maxlen=temporal_window)
        
        center_x = (initial_bbox[0] + initial_bbox[2]) / 2
        center_y = (initial_bbox[1] + initial_bbox[3]) / 2
        self.position_history.append((center_x, center_y))
        
    def update(self, bbox):
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        if len(self.position_history) > 0:
            last_x, last_y = self.position_history[-1]
            movement = math.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
            self.movement_history.append(movement)
        else:
            self.movement_history.append(0)
            
        self.position_history.append((center_x, center_y))
        
    def get_movement_score(self):
        if len(self.movement_history) == 0:
            return 0
        return np.mean(self.movement_history)
    
    def is_moving(self, threshold=15.0):
        return self.get_movement_score() > threshold

class TemporalAnalyzer:
    def __init__(self, temporal_window=10, movement_threshold=15.0):
        self.temporal_window = temporal_window
        self.movement_threshold = movement_threshold
        self.patient_trackers = {}
        self.frame_count = 0
        
    def update_tracks(self, detections, track_ids):
        current_patients = set()
        
        for detection, track_id in zip(detections, track_ids):
            if track_id not in self.patient_trackers:
                self.patient_trackers[track_id] = SimplePatientTracker(
                    track_id, detection, self.temporal_window
                )
            else:
                self.patient_trackers[track_id].update(detection)
            
            current_patients.add(track_id)
        
        expired_tracks = set(self.patient_trackers.keys()) - current_patients
        for track_id in expired_tracks:
            del self.patient_trackers[track_id]
        
        self.frame_count += 1
        
    def analyze_room_activity(self):
        if not self.patient_trackers:
            return {
                'patient_count': 0,
                'moving_patients': 0,
                'status': 'EMPTY',
                'warning': False
            }
        
        patient_count = len(self.patient_trackers)
        moving_patients = sum(1 for tracker in self.patient_trackers.values() 
                            if tracker.is_moving(self.movement_threshold))
        
        if patient_count == 0:
            status = 'EMPTY'
            warning = False
        elif patient_count == 1:
            main_patient = next(iter(self.patient_trackers.values()))
            if main_patient.is_moving(self.movement_threshold):
                status = 'SINGLE_PATIENT_MOVING'
                warning = True
            else:
                status = 'SINGLE_PATIENT_IDLE'
                warning = False
        else:
            status = 'MULTIPLE_PEOPLE'
            warning = False
        
        return {
            'patient_count': patient_count,
            'moving_patients': moving_patients,
            'status': status,
            'warning': warning
        }
    
    def get_patient_movements(self):
        movements = {}
        for track_id, tracker in self.patient_trackers.items():
            movements[track_id] = {
                'movement_score': tracker.get_movement_score(),
                'is_moving': tracker.is_moving(self.movement_threshold)
            }
        return movements

# Custom annotation functions
def draw_bounding_box(image, bbox, color, label, thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0] + 5, y1), color, -1)
    
    # Draw label text
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.WHITE, 1)
    
    return image

# Main Patient Monitor Class
class HospitalPatientMonitor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.temporal_analyzer = TemporalAnalyzer(
            temporal_window=config.TEMPORAL_WINDOW,
            movement_threshold=config.MOVEMENT_THRESHOLD
        )
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize YOLO model with proper error handling"""
        if not YOLO_AVAILABLE:
            st.warning("🔄 YOLO is not available. Running in demo mode with simulated detection.")
            return
        
        try:
            st.info("🔄 Loading YOLO model... This may take a moment.")
            
            # Force weights_only=False for PyTorch 2.6+ compatibility
            import torch
            original_load = torch.load
            
            def custom_load(f, map_location=None, pickle_module=None, *, weights_only=False, **kwargs):
                return original_load(f, map_location, pickle_module, weights_only=False, **kwargs)
            
            torch.load = custom_load
            
            # Load YOLO model
            self.model = YOLO(self.config.MODEL_NAME)
            st.success("✅ YOLOv8 model loaded successfully!")
            
            # Test with a small dummy inference
            dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_frame, verbose=False)
            st.success("✅ Model tested and ready!")
            
        except Exception as e:
            st.warning(f"⚠️ YOLO loading failed: {e}")
            st.info("🔄 Running in demo mode with simulated patient detection.")
            self.model = None
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame for patient detection and analysis"""
        if self.model is None:
            return self._demo_mode(frame)
        
        try:
            # Run YOLO inference without tracking (to avoid lap dependency)
            results = self.model(
                frame,
                conf=self.config.CONFIDENCE_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                verbose=False
            )
            
            if not results or len(results) == 0:
                room_analysis = self.temporal_analyzer.analyze_room_activity()
                return self._annotate_frame(frame, [], [], room_analysis), room_analysis
            
            result = results[0]
            person_detections = []
            track_ids = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    if int(box.cls) == self.config.PERSON_CLASS_ID and box.conf > self.config.CONFIDENCE_THRESHOLD:
                        bbox = box.xyxy[0].cpu().numpy()
                        person_detections.append(bbox)
                        track_ids.append(i)  # Use index as ID since we're not tracking
            
            if len(person_detections) > 0:
                self.temporal_analyzer.update_tracks(person_detections, track_ids)
            
            room_analysis = self.temporal_analyzer.analyze_room_activity()
            annotated_frame = self._annotate_frame(frame, person_detections, track_ids, room_analysis)
            
            return annotated_frame, room_analysis
            
        except Exception as e:
            st.error(f"❌ Error in process_frame: {e}")
            # Fall back to demo mode on error
            return self._demo_mode(frame)
    
    def _demo_mode(self, frame: np.ndarray) -> tuple:
        """Demo mode with simulated patient detection"""
        h, w = frame.shape[:2]
        
        # Simulate patient detection - create a bounding box that moves slightly
        demo_detections = []
        demo_track_ids = []
        
        # Add some random movement to simulate real detection
        movement_offset = int(5 * math.sin(time.time()))
        
        # Create a demo patient bounding box
        demo_bbox = [
            w//4 + movement_offset, 
            h//4 + movement_offset, 
            w//4 + 150 + movement_offset, 
            h//4 + 300 + movement_offset
        ]
        demo_detections.append(demo_bbox)
        demo_track_ids.append(1)
        
        # Occasionally add a second "staff" person
        if int(time.time()) % 10 < 3:  # 30% of the time
            staff_bbox = [
                w//2, h//3, 
                w//2 + 120, h//3 + 250
            ]
            demo_detections.append(staff_bbox)
            demo_track_ids.append(2)
        
        # Update temporal analyzer with demo data
        self.temporal_analyzer.update_tracks(demo_detections, demo_track_ids)
        
        room_analysis = self.temporal_analyzer.analyze_room_activity()
        annotated_frame = self._annotate_frame(frame, demo_detections, demo_track_ids, room_analysis)
        
        return annotated_frame, room_analysis
    
    def _annotate_frame(self, frame: np.ndarray, detections, track_ids, room_analysis) -> np.ndarray:
        """Annotate frame with bounding boxes and status information"""
        annotated_frame = frame.copy()
        
        status = room_analysis['status']
        warning = room_analysis['warning']
        
        # Status text with color coding
        if warning:
            color = Colors.RED
            status_text = f"WARNING: {status}"
        elif status == 'EMPTY':
            color = Colors.YELLOW
            status_text = f"Status: {status}"
        else:
            color = Colors.GREEN
            status_text = f"Status: {status}"
        
        # Add status overlay
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        count_text = f"Patients: {room_analysis['patient_count']}"
        cv2.putText(annotated_frame, count_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.WHITE, 2)
        
        moving_text = f"Moving: {room_analysis.get('moving_patients', 0)}"
        cv2.putText(annotated_frame, moving_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.WHITE, 2)
        
        # Add mode indicator
        mode_text = "DEMO MODE" if self.model is None else "YOLO MODE"
        mode_color = Colors.YELLOW if self.model is None else Colors.GREEN
        cv2.putText(annotated_frame, mode_text, (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # Annotate detections
        if len(detections) > 0:
            movements = self.temporal_analyzer.get_patient_movements()
            
            for i, bbox in enumerate(detections):
                track_id = track_ids[i] if i < len(track_ids) else i
                
                # Determine label and color
                if track_id in movements:
                    movement_status = "MOVING" if movements[track_id]['is_moving'] else "IDLE"
                    label = f"P{track_id} {movement_status}"
                    bbox_color = Colors.RED if movements[track_id]['is_moving'] else Colors.GREEN
                else:
                    label = f"P{track_id}"
                    bbox_color = Colors.BLUE
                
                annotated_frame = draw_bounding_box(annotated_frame, bbox, bbox_color, label)
        
        # Add warning overlay if needed
        if warning:
            h, w = annotated_frame.shape[:2]
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), Colors.RED, -1)
            cv2.addWeighted(overlay, 0.1, annotated_frame, 0.9, 0, annotated_frame)
            
            warning_text = "ALERT: Single patient moving alone!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(annotated_frame, warning_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, Colors.RED, 2)
        
        return annotated_frame

# Streamlit App
class HospitalMonitoringApp:
    def __init__(self):
        self.model_config = ModelConfig()
        self.app_config = AppConfig()
        self.monitor = HospitalPatientMonitor(self.model_config)
        
    def process_webcam(self):
        st.header("📹 Live Webcam Monitoring")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Start Webcam Monitoring", use_container_width=True, key="start_webcam"):
                self.run_webcam()
        
        with col2:
            if st.button("🔄 Reset Detection", use_container_width=True, key="reset_webcam"):
                st.rerun()
        
        st.info("💡 **Instructions**: Click 'Start Webcam Monitoring' to begin real-time patient detection using your webcam.")
        
        # Display mode information
        if self.monitor.model is None:
            st.warning("🔸 **Running in Demo Mode** - Showing simulated patient detection")
        else:
            st.success("🔹 **Running in YOLO Mode** - Real person detection active")
    
    def run_webcam(self):
        """Run webcam monitoring"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access webcam. Please check if it's connected.")
                return
            
            # Set camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            stframe = st.empty()
            status_placeholder = st.empty()
            warning_placeholder = st.empty()
            
            stop_button = st.button("🛑 Stop Monitoring", key="stop_webcam")
            
            st.success("🔴 Live monitoring started! Press 'Stop Monitoring' to end.")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to capture frame from webcam")
                    break
                
                # Process frame
                processed_frame, analysis = self.monitor.process_frame(frame)
                
                # Convert BGR to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display processed frame
                stframe.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                
                # Display analysis results
                self.display_analysis(analysis, status_placeholder, warning_placeholder)
                
                # Small delay to prevent high CPU usage
                time.sleep(0.1)
                
            cap.release()
            st.success("✅ Webcam monitoring stopped")
            
        except Exception as e:
            st.error(f"❌ Error in webcam processing: {e}")
    
    def process_video(self):
        st.header("🎥 Video File Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze patient movements"
        )
        
        if uploaded_file is not None:
            st.info(f"📁 **File uploaded**: {uploaded_file.name}")
            
            # Display file info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.write(f"📊 File size: {file_size:.2f} MB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Analyze Video", use_container_width=True, key="analyze_video"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        video_path = tmp_file.name
                    
                    self.analyze_video_file(video_path)
                    
                    try:
                        os.unlink(video_path)
                    except:
                        pass
            
            with col2:
                if st.button("🔄 Clear Video", use_container_width=True, key="clear_video"):
                    st.rerun()
    
    def analyze_video_file(self, video_path):
        """Analyze uploaded video file"""
        try:
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            video_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                st.error("❌ Could not read video file. Please try a different video.")
                return
            
            st.info(f"🎬 Video info: {total_frames} frames, {fps:.1f} FPS")
            
            frame_count = 0
            analysis_results = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, analysis = self.monitor.process_frame(frame)
                analysis_results.append(analysis)
                
                # Convert for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                
                # Display current analysis
                self.display_analysis(analysis, status_placeholder)
                
                # Update progress
                if total_frames > 0:
                    progress = (frame_count + 1) / total_frames
                    progress_bar.progress(min(progress, 1.0))
                
                frame_count += 1
                
                # Show processing stats
                if frame_count % 30 == 0:
                    stats_placeholder.info(f"🔄 Processed {frame_count}/{total_frames} frames...")
            
            cap.release()
            
            # Show final statistics
            self.show_video_statistics(analysis_results)
            st.success("✅ Video analysis completed!")
            
        except Exception as e:
            st.error(f"❌ Error processing video: {e}")
    
    def show_video_statistics(self, analysis_results):
        """Display video analysis statistics"""
        st.header("📊 Video Analysis Summary")
        
        total_frames = len(analysis_results)
        warning_frames = sum(1 for analysis in analysis_results if analysis.get('warning', False))
        patient_frames = sum(1 for analysis in analysis_results if analysis.get('patient_count', 0) > 0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Frames", total_frames)
        
        with col2:
            st.metric("Warning Frames", warning_frames)
        
        with col3:
            st.metric("Patient Detected Frames", patient_frames)
        
        with col4:
            warning_percentage = (warning_frames / total_frames * 100) if total_frames > 0 else 0
            st.metric("Warning Percentage", f"{warning_percentage:.1f}%")
        
        # Show timeline of patient count
        patient_counts = [analysis.get('patient_count', 0) for analysis in analysis_results]
        frames = list(range(len(patient_counts)))
        
        # Simple text-based timeline
        st.subheader("Patient Count Timeline")
        timeline_text = "".join(["●" if count > 0 else "○" for count in patient_counts[:100]])  # First 100 frames
        st.text(timeline_text)
        st.caption("● = Patients detected, ○ = No patients")
    
    def process_image(self):
        st.header("🖼️ Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to test patient detection"
        )
        
        if uploaded_file is not None:
            st.info(f"📁 **File uploaded**: {uploaded_file.name}")
            
            # Display original image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                if st.button("🔍 Analyze Image", use_container_width=True, key="analyze_image"):
                    try:
                        # Convert to numpy array
                        image_np = np.array(image)
                        
                        # Convert to BGR for processing
                        if image_np.shape[-1] == 4:
                            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
                        else:
                            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        
                        # Process image
                        processed_image, analysis = self.monitor.process_frame(image_bgr)
                        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        
                        # Display results
                        st.subheader("Processed Image")
                        st.image(processed_image_rgb, use_column_width=True)
                        
                        # Display analysis
                        self.display_analysis(analysis)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing image: {e}")
            
            # Clear button
            if st.button("🔄 Clear Image", use_container_width=True, key="clear_image"):
                st.rerun()
    
    def display_analysis(self, analysis, status_placeholder=None, warning_placeholder=None):
        """Display analysis results"""
        if status_placeholder is None:
            status_placeholder = st
        
        # Create metrics columns
        col1, col2, col3, col4 = status_placeholder.columns(4)
        
        with col1:
            st.metric("Patient Count", analysis.get('patient_count', 0))
        
        with col2:
            st.metric("Moving Patients", analysis.get('moving_patients', 0))
        
        with col3:
            status = analysis.get('status', 'UNKNOWN')
            status_color = "🟢" if not analysis.get('warning', False) else "🔴"
            st.metric("Room Status", f"{status_color} {status}")
        
        with col4:
            warning_status = "ACTIVE" if analysis.get('warning', False) else "INACTIVE"
            st.metric("Warning System", warning_status)
        
        # Display warning message if needed
        if warning_placeholder is not None:
            if analysis.get('warning', False):
                warning_placeholder.error("""
                🚨 **ALERT: Single patient detected moving alone!**
                
                **Action Required:** Check patient immediately
                """)
            else:
                warning_placeholder.success(f"""
                ✅ **ALL CLEAR**
                
                Room status: **{analysis.get('status', 'UNKNOWN').replace('_', ' ').title()}**
                
                No immediate action required
                """)
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🏥 Hospital Patient Monitoring System</h1>', unsafe_allow_html=True)
        st.markdown("### Advanced YOLOv8 + Temporal Analysis for Patient Safety Monitoring")
        
        # System status
        if self.monitor.model is None:
            st.warning("🔸 **System Status**: Running in **Demo Mode** (YOLO tracking not available)")
        else:
            st.success("🔹 **System Status**: Running in **YOLO Mode** (Real-time detection active)")
        
        # Sidebar navigation with clear options
        st.sidebar.title("🎯 Monitoring Options")
        app_mode = st.sidebar.radio(
            "Choose Input Source:",
            ["Webcam Monitoring", "Video Analysis", "Image Analysis", "About System"]
        )
        
        # Route to appropriate mode
        if app_mode == "Webcam Monitoring":
            self.process_webcam()
        elif app_mode == "Video Analysis":
            self.process_video()
        elif app_mode == "Image Analysis":
            self.process_image()
        else:
            self.show_about()
    
    def show_about(self):
        """Show about section"""
        st.header("About the System")
        
        st.markdown("""
        ### 🏥 Hospital Patient Monitoring System
        
        **Novel Implementation using YOLOv8 with Temporal Analysis**
        
        This advanced system provides intelligent patient monitoring with real-time alerts using computer vision and temporal analysis.
        
        #### 🎯 Input Options:
        - **📹 Webcam Monitoring**: Real-time detection from your webcam
        - **🎥 Video Analysis**: Upload and analyze video files
        - **🖼️ Image Analysis**: Upload and analyze single images
        
        #### 🔧 Technical Features:
        - **YOLOv8 Object Detection**: State-of-the-art person detection
        - **Temporal Movement Analysis**: Track movements over time
        - **Smart Alert System**: Only warns when single patient moves alone
        - **Multi-person Safety**: No alerts when staff is present
        
        #### 📊 Detection Logic:
        - **Empty Room** → No alerts
        - **Single Patient Moving** → 🚨 ALERT (Check immediately)
        - **Single Patient Idle** → ✅ Safe (No action needed)
        - **Multiple People** → ✅ Safe (Staff present)
        
        #### 🎮 Demo Mode:
        - Simulated patient detection when YOLO tracking unavailable
        - Demonstrates all system features
        - Visual feedback with color-coded bounding boxes
        """)
        
        # System requirements
        st.sidebar.info("""
        **💻 System Requirements:**
        - Webcam for live monitoring
        - Supported video formats: MP4, AVI, MOV, MKV
        - Supported image formats: JPG, JPEG, PNG
        - Internet connection for YOLO model download
        """)

# Main execution
if __name__ == "__main__":
    # Page configuration
    st.set_page_config(
        page_title="Hospital Patient Monitoring System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stButton button {
            width: 100%;
            margin: 5px 0;
        }
        .uploadedFile {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and run the app
    app = HospitalMonitoringApp()
    app.run()