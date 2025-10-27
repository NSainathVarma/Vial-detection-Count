import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Vial Detection Chatbot",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

class VialDetectionChatbot:
    def __init__(self, model_path="best.pt"):
        # Setup LLM
        with open("api_key.txt", "r") as f:
            api_key = f.read().strip()
        os.environ["GOOGLE_API_KEY"] = api_key
        
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            st.sidebar.success("✅ Vial detection model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load model: {e}")
            self.model = None
    
    def preprocess_image(self, image):
        """
        Convert any image format to YOLO-compatible format
        """
        try:
            # If it's a file path
            if isinstance(image, str):
                pil_image = Image.open(image)
            # If it's a PIL Image
            elif isinstance(image, Image.Image):
                pil_image = image
            # If it's a file upload object
            elif hasattr(image, 'read'):
                pil_image = Image.open(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            return pil_image
            
        except Exception as e:
            st.error(f"Image preprocessing error: {e}")
            return None
    
    def detect_vials(self, image, confidence_threshold=0.6):
        """
        Detect vials in image and return annotated image + results
        Only shows bounding boxes above confidence threshold
        """
        if self.model is None:
            st.error("Model not loaded properly!")
            return image, 0, 0, [], 0
            
        try:
            # Preprocess image
            pil_image = self.preprocess_image(image)
            if pil_image is None:
                return image, 0, 0, [], 0
            
            # Convert PIL to numpy array for YOLO
            image_np = np.array(pil_image)
            
            # Run inference
            results = self.model(image_np)
            
            annotated_image = pil_image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                except:
                    font = ImageFont.load_default()
            
            detections = []
            vial_count = 0
            total_confidence = 0
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = box.cls[0].cpu().numpy()
                        label = self.model.names[int(class_id)]
                        
                        # Only process detections above confidence threshold
                        if confidence >= confidence_threshold:
                            detection_info = {
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "confidence": float(confidence),
                                "label": label
                            }
                            detections.append(detection_info)
                            
                            if 'vial' in label.lower():
                                vial_count += 1
                                total_confidence += confidence
                            
                            # Draw bounding box (only for high confidence detections)
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            
                            # Draw label
                            label_text = f"{label} {confidence:.2f}"
                            # Use textbbox for newer PIL versions
                            try:
                                text_bbox = draw.textbbox((x1, y1-25), label_text, font=font)
                            except:
                                # Fallback for older PIL versions
                                text_width = draw.textlength(label_text, font=font)
                                text_bbox = (x1, y1-25, x1 + text_width, y1-5)
                            
                            draw.rectangle(text_bbox, fill="red")
                            draw.text((x1, y1-25), label_text, fill="white", font=font)
            
            # Add summary to image
            avg_confidence = total_confidence / vial_count if vial_count > 0 else 0
            summary_text = f"Vials: {vial_count} | Total: {len(detections)}"
            summary_text += f" | Threshold: {confidence_threshold}"
            if vial_count > 0:
                summary_text += f" | Avg Conf: {avg_confidence:.2f}"
            
            try:
                summary_bbox = draw.textbbox((10, 10), summary_text, font=font)
            except:
                summary_width = draw.textlength(summary_text, font=font)
                summary_bbox = (10, 10, 10 + summary_width, 35)
            
            draw.rectangle(summary_bbox, fill="green")
            draw.text((10, 10), summary_text, fill="white", font=font)
            
            return annotated_image, len(detections), vial_count, detections, avg_confidence
            
        except Exception as e:
            st.error(f"Detection error: {e}")
            # Return original image if detection fails
            return pil_image, 0, 0, [], 0

    def process_message(self, user_input, image=None):
        """
        Process user message with optional image
        """
        try:
            context = "You are a helpful vial detection assistant. You can chat normally or analyze images for vials.\n\n"
            
            if image is not None:
                # Perform detection
                annotated_img, total_objects, vial_count, detections, avg_confidence = self.detect_vials(image)
                
                # Add detection results to context
                context += f"Vial Detection Results:\n"
                context += f"- Total objects detected: {total_objects}\n"
                context += f"- Vials identified: {vial_count}\n"
                if vial_count > 0:
                    context += f"- Average confidence: {avg_confidence:.3f}\n"
                
                if detections:
                    context += "\nDetailed detections:\n"
                    vial_detections = [d for d in detections if 'vial' in d['label'].lower()]
                    other_detections = [d for d in detections if 'vial' not in d['label'].lower()]
                    
                    if vial_detections:
                        context += "Vials:\n"
                        for i, det in enumerate(vial_detections, 1):
                            context += f"{i}. {det['label']} (Confidence: {det['confidence']:.3f})\n"
                    
                    if other_detections:
                        context += "Other objects:\n"
                        for i, det in enumerate(other_detections, 1):
                            context += f"{i}. {det['label']} (Confidence: {det['confidence']:.3f})\n"
                
                context += f"\nUser question: {user_input}\n"
                context += "\nPlease provide a helpful response explaining these detection results to the user."
                
                # Get LLM response
                response = self.llm.invoke(context)
                
                return response.content, annotated_img, vial_count, total_objects, detections
            else:
                # Regular chat without image
                context += f"User: {user_input}"
                response = self.llm.invoke(context)
                return response.content, None, 0, 0, []
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}", None, 0, 0, []

# Initialize chatbot
@st.cache_resource
def load_chatbot():
    return VialDetectionChatbot("best.pt")

def main():
    # Sidebar
    st.sidebar.title("🧪 Vial Detection Chatbot")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### How to use:")
    st.sidebar.markdown("""
    1. 📸 **Upload an image** (JPEG, PNG, BMP)
    2. 💬 **Ask a question** about the image
    3. 🎯 **View results** with bounding boxes
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Supported Image Formats:")
    st.sidebar.markdown("""
    - JPEG/JPG
    - PNG
    - BMP
    - WebP
    """)
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vial_bot" not in st.session_state:
        st.session_state.vial_bot = load_chatbot()
    
    # Main content
    st.title("🧪 Vial Detection Chatbot")
    st.markdown("Upload an image to detect vials or chat with me about vial detection!")
    
    # File uploader with more specific instructions
    uploaded_file = st.file_uploader(
        "📸 Upload an image for vial detection", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
        help="Supported formats: JPG, PNG, BMP, WebP"
    )
    
    # Display uploaded image with validation
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"📤 Uploaded Image - {uploaded_file.name}", width=400)
            st.success(f"✅ Image loaded successfully: {image.size[0]}x{image.size[1]} pixels, {image.mode} mode")
        except Exception as e:
            st.error(f"❌ Failed to load image: {e}")
            uploaded_file = None
    
    # Chat input
    user_input = st.chat_input("💬 Ask about vials or upload an image...")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.write(message["content"])
            elif message["type"] == "image":
                st.image(message["content"], use_container_width=True)
            elif message["type"] == "stats":
                cols = st.columns(3)
                cols[0].metric("Vials Detected", message["vial_count"])
                cols[1].metric("Total Objects", message["total_objects"])
                if message["vial_count"] > 0:
                    cols[2].metric("Avg Confidence", f"{message['avg_confidence']:.2f}")
    
    # Process user input
    if user_input:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(user_input)
        
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input, 
            "type": "text"
        })
        
        # Process message
        with st.spinner("🔍 Analyzing image..." if uploaded_file else "💭 Thinking..."):
            if uploaded_file is not None:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                # Process with image
                response, annotated_image, vial_count, total_objects, detections = st.session_state.vial_bot.process_message(
                    user_input, uploaded_file
                )
            else:
                # Process without image
                response, annotated_image, vial_count, total_objects, detections = st.session_state.vial_bot.process_message(user_input)
        
        # Display assistant response
        with st.chat_message("assistant"):
            if annotated_image is not None:
                # Show detection statistics
                if vial_count > 0 or total_objects > 0:
                    cols = st.columns(3)
                    cols[0].metric("🧪 Vials Detected", vial_count)
                    cols[1].metric("📊 Total Objects", total_objects)
                    if vial_count > 0:
                        avg_conf = sum(d['confidence'] for d in detections if 'vial' in d['label'].lower()) / vial_count
                        cols[2].metric("🎯 Avg Confidence", f"{avg_conf:.2f}")
                
                # Show annotated image
                st.image(annotated_image, caption="🎯 Detection Results", width=500)
                
                # Show detailed detections
                if detections:
                    with st.expander("📋 View Detailed Detection Results"):
                        vial_detections = [d for d in detections if 'vial' in d['label'].lower()]
                        other_detections = [d for d in detections if 'vial' not in d['label'].lower()]
                        
                        if vial_detections:
                            st.subheader("🧪 Vials Found:")
                            for i, detection in enumerate(vial_detections, 1):
                                st.write(f"**{i}. {detection['label']}**")
                                st.write(f"   Confidence: `{detection['confidence']:.3f}`")
                                st.write(f"   Position: `{detection['bbox']}`")
                                st.write("---")
                        
                        if other_detections:
                            st.subheader("📦 Other Objects:")
                            for i, detection in enumerate(other_detections, 1):
                                st.write(f"**{i}. {detection['label']}**")
                                st.write(f"   Confidence: `{detection['confidence']:.3f}`")
                                st.write("---")
            
            # Show text response
            st.write(response)
        
        # Add to chat history
        if annotated_image is not None:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": annotated_image,
                "type": "image"
            })
            
            # Add stats to history
            avg_conf = sum(d['confidence'] for d in detections if 'vial' in d['label'].lower()) / vial_count if vial_count > 0 else 0
            st.session_state.chat_history.append({
                "role": "assistant",
                "type": "stats",
                "vial_count": vial_count,
                "total_objects": total_objects,
                "avg_confidence": avg_conf
            })
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "type": "text"
        })

if __name__ == "__main__":
    main()