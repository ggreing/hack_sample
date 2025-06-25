import streamlit as st
import cv2
import numpy as np
import time
import json
from threading import Thread, Lock
import queue
import mediapipe as mp
import dlib
import face_recognition
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import logging
import socketio
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AttendanceSystem')

# 서버 설정
SERVER_URL = "http://13.238.227.125:3000"

# SocketIO 클라이언트 클래스
class SocketIOClient:
    def __init__(self):
        self.sio = socketio.Client()
        self.connected = False
        self.setup_events()
        
    def setup_events(self):
        @self.sio.event
        def connect():
            logger.info("SocketIO 서버에 연결되었습니다")
            self.connected = True
            
        @self.sio.event
        def disconnect():
            logger.info("SocketIO 서버에서 연결이 해제되었습니다")
            self.connected = False
            
        @self.sio.on('attendance_response')
        def on_attendance_response(data):
            logger.info(f"서버 응답: {data}")
            
    def connect_to_server(self):
        try:
            if not self.connected:
                self.sio.connect(SERVER_URL, transports=['websocket'])
                return True
        except Exception as e:
            logger.error(f"서버 연결 실패: {e}")
            return False
        return self.connected
    
    def send_attendance_data(self, student_data, analysis_result):
        """출석 데이터를 서버로 전송"""
        if self.connected:
            try:
                payload = {
                    'student_id': student_data.get('id'),
                    'student_name': student_data.get('name'),
                    'timestamp': time.time(),
                    'analysis_result': analysis_result,
                    'endpoint': '/api/attendance'
                }
                self.sio.emit('attendance_update', payload)
                logger.info(f"출석 데이터 전송 완료: {student_data.get('name')}")
                return True
            except Exception as e:
                logger.error(f"데이터 전송 실패: {e}")
                return False
        return False
    
    def disconnect_from_server(self):
        if self.connected:
            self.sio.disconnect()

# 페이지 설정
st.set_page_config(
    page_title="실시간 온라인 강의 출석 시스템",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AttendanceAnalyzer:
    def __init__(self, criteria_weights=None):
        try:
            # MediaPipe 초기화
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # 판단 기준 가중치 설정 (기본값)
            self.criteria_weights = criteria_weights or {
                'face_detected': 0.2,
                'face_recognized': 0.3,
                'liveness': 0.15,
                'attention': 0.25,
                'head_pose': 0.1
            }
            
            # 분석 변수
            self.face_detected = False
            self.face_recognized = False
            self.blink_count = 0
            self.last_blink_time = time.time()
            self.head_pose_angles = {'yaw': 0, 'pitch': 0, 'roll': 0}
            self.attention_score = 0.0
            
            # 스레드 안전성을 위한 Lock
            self.analysis_lock = Lock()
            
            # 3D 모델 포인트 (PnP 솔버용)
            self.model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ], dtype=np.float32)
            
            # 카메라 매트릭스
            self.focal_length = 600.0
            self.camera_matrix = np.array([
                [self.focal_length, 0, 320],
                [0, self.focal_length, 240],
                [0, 0, 1]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros((4, 1))
            
            logger.info("AttendanceAnalyzer 초기화 완료")
        except Exception as e:
            logger.error(f"AttendanceAnalyzer 초기화 실패: {e}")
            raise e
    
    def update_criteria_weights(self, new_weights):
        """판단 기준 가중치 업데이트"""
        with self.analysis_lock:
            self.criteria_weights = new_weights
            logger.info(f"판단 기준 가중치 업데이트: {new_weights}")
    
    def calculate_ear(self, landmarks, eye_points):
        try:
            eye = np.array([(landmarks[point].x, landmarks[point].y) for point in eye_points])
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            ear = (A + B) / (2.0 * C) if C != 0 else 0.3
            return ear
        except Exception as e:
            logger.error(f"EAR 계산 오류: {e}")
            return 0.3
    
    def detect_blink(self, landmarks):
        left_eye_points = [33, 160, 158, 133, 153, 144]
        right_eye_points = [362, 385, 387, 263, 373, 380]
        
        left_ear = self.calculate_ear(landmarks, left_eye_points)
        right_ear = self.calculate_ear(landmarks, right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        blink_threshold = 0.27
        current_time = time.time()
        
        if avg_ear < blink_threshold and (current_time - self.last_blink_time) > 0.3:
            self.last_blink_time = current_time
            return True
        return False
    
    def estimate_head_pose(self, landmarks, frame_shape):
        try:
            if len(landmarks) < 468:
                logger.warning("랜드마크 개수 부족")
                return
                
            image_points = np.array([
                (landmarks[1].x * frame_shape[1], landmarks[1].y * frame_shape[0]),
                (landmarks[152].x * frame_shape[1], landmarks[152].y * frame_shape[0]),
                (landmarks[226].x * frame_shape[1], landmarks[226].y * frame_shape[0]),
                (landmarks[446].x * frame_shape[1], landmarks[446].y * frame_shape[0]),
                (landmarks[57].x * frame_shape[1], landmarks[57].y * frame_shape[0]),
                (landmarks[287].x * frame_shape[1], landmarks[287].y * frame_shape[0])
            ], dtype=np.float32)
            
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix, self.dist_coeffs
            )
            
            if success:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
                
                singular = sy < 1e-6
                if not singular:
                    yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
                    pitch = np.arctan2(-rotation_matrix[2,0], sy)
                    roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                else:
                    yaw = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                    pitch = np.arctan2(-rotation_matrix[2,0], sy)
                    roll = 0
                
                with self.analysis_lock:
                    self.head_pose_angles = {
                        'yaw': np.degrees(yaw),
                        'pitch': np.degrees(pitch),
                        'roll': np.degrees(roll)
                    }
        except Exception as e:
            logger.error(f"머리 자세 추정 중 오류: {e}")
    
    def analyze_frame(self, frame):
        """프레임 분석"""
        try:
            if frame is None or frame.size == 0:
                logger.warning("빈 프레임 수신")
                return self._get_default_result()
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            face_detected = False
            is_alive = False
            attention_focused = False
            
            if results.multi_face_landmarks:
                face_detected = True
                
                for face_landmarks in results.multi_face_landmarks:
                    current_time = time.time()
                    if self.detect_blink(face_landmarks.landmark):
                        with self.analysis_lock:
                            self.blink_count += 1
                            self.last_blink_time = current_time
                    
                    with self.analysis_lock:
                        is_alive = (current_time - self.last_blink_time) < 2.0
                    
                    self.estimate_head_pose(face_landmarks.landmark, frame.shape)
                    
                    with self.analysis_lock:
                        yaw_ok = abs(self.head_pose_angles['yaw']) < 30
                        pitch_ok = abs(self.head_pose_angles['pitch']) < 40
                        roll_val = self.head_pose_angles['roll']
                        roll_ok = abs(roll_val) < 25 or abs(abs(roll_val) - 180) < 25
                        attention_focused = yaw_ok and pitch_ok and roll_ok
            
            # 동적 점수 계산 (설정된 가중치 사용)
            with self.analysis_lock:
                score_components = {
                    'face_detected': self.criteria_weights['face_detected'] if face_detected else 0.0,
                    'face_recognized': self.criteria_weights['face_recognized'] if face_detected else 0.0,
                    'liveness': self.criteria_weights['liveness'] if is_alive else 0.0,
                    'attention': self.criteria_weights['attention'] if attention_focused else 0.0,
                    'head_pose': self.criteria_weights['head_pose'] if attention_focused else 0.0
                }
            
            total_score = sum(score_components.values())
            
            # 출석 상태 결정
            if total_score >= 0.8:
                status = "excellent"
            elif total_score >= 0.7:
                status = "present"
            elif total_score >= 0.5:
                status = "attention_needed"
            else:
                status = "absent"
            
            with self.analysis_lock:
                return {
                    'face_detected': face_detected,
                    'face_recognized': face_detected,
                    'is_alive': is_alive,
                    'attention_focused': attention_focused,
                    'head_pose': self.head_pose_angles.copy(),
                    'blink_count': self.blink_count,
                    'attendance_score': total_score,
                    'attendance_status': status,
                    'score_components': score_components,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.error(f"프레임 분석 중 오류: {e}")
            return self._get_default_result()
    
    def _get_default_result(self):
        """오류 발생 시 기본값 반환"""
        with self.analysis_lock:
            return {
                'face_detected': False,
                'face_recognized': False,
                'is_alive': False,
                'attention_focused': False,
                'head_pose': self.head_pose_angles.copy(),
                'blink_count': self.blink_count,
                'attendance_score': 0.0,
                'attendance_status': "error",
                'score_components': {},
                'timestamp': time.time()
            }

# WebRTC 비디오 프로세서 클래스
class VideoProcessor:
    def __init__(self, criteria_weights=None):
        self.frame_count = 0
        self.analysis_interval = 5
        
        try:
            self.analyzer = AttendanceAnalyzer(criteria_weights)
            self.last_analysis = None
            self.analysis_lock = Lock()
            logger.info("VideoProcessor 초기화 완료")
        except Exception as e:
            logger.error(f"VideoProcessor 초기화 실패: {e}")
            self.analyzer = None
    
    def update_criteria_weights(self, new_weights):
        """판단 기준 가중치 업데이트"""
        if self.analyzer:
            self.analyzer.update_criteria_weights(new_weights)
    
    def recv(self, frame):
        try:
            if self.analyzer is None:
                logger.error("분석기가 초기화되지 않음")
                return frame
                
            img = frame.to_ndarray(format="bgr24")
            
            if self.frame_count % self.analysis_interval == 0:
                analysis_result = self.analyzer.analyze_frame(img)
                
                with self.analysis_lock:
                    self.last_analysis = analysis_result
            
            self.frame_count += 1
            self.draw_analysis_result(img)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            logger.error(f"프레임 처리 중 예외 발생: {e}")
            return frame
    
    def draw_analysis_result(self, frame):
        """분석 결과를 프레임에 표시"""
        try:
            with self.analysis_lock:
                if self.last_analysis:
                    result = self.last_analysis
                    
                    status_colors = {
                        "excellent": (0, 255, 0),
                        "present": (0, 200, 255),
                        "attention_needed": (0, 165, 255),
                        "absent": (0, 0, 255),
                        "error": (255, 0, 255)
                    }
                    
                    status = result['attendance_status']
                    color = status_colors.get(status, (255, 255, 255))
                    
                    cv2.putText(frame, f"Status: {status.upper()}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Score: {result['attendance_score']:.2f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Face: {'✓' if result['face_detected'] else '✗'}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if result['face_detected'] else (0, 0, 255), 2)
                    cv2.putText(frame, f"Alive: {'✓' if result['is_alive'] else '✗'}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if result['is_alive'] else (0, 0, 255), 2)
                    cv2.putText(frame, f"Attention: {'✓' if result['attention_focused'] else '✗'}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if result['attention_focused'] else (0, 0, 255), 2)
                    
        except Exception as e:
            logger.error(f"결과 표시 중 오류: {e}")
            cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def get_last_analysis(self):
        """마지막 분석 결과 반환"""
        with self.analysis_lock:
            if self.last_analysis:
                return self.last_analysis.copy()
            else:
                return {
                    'face_detected': False,
                    'face_recognized': False,
                    'is_alive': False,
                    'attention_focused': False,
                    'head_pose': {'yaw': 0, 'pitch': 0, 'roll': 0},
                    'blink_count': 0,
                    'attendance_score': 0.0,
                    'attendance_status': "error",
                    'score_components': {},
                    'timestamp': time.time()
                }

# 메인 애플리케이션
def main():
        # 세션 상태 초기화
    if 'socketio_client' not in st.session_state:
        st.session_state.socketio_client = SocketIOClient()
    
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = None
        
        
    st.title("🎓 실시간 온라인 강의 출석 시스템")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    
    # 학생 정보 입력
    student_id = st.sidebar.text_input("학생 ID", value="2024001")
    student_name = st.sidebar.text_input("학생 이름", value="김철수")
    
    # 서버 연결 상태
    st.sidebar.subheader("🌐 서버 연결")
    connection_status = st.sidebar.empty()
    
    if st.sidebar.button("서버 연결"):
        if st.session_state.socketio_client.connect_to_server():
            st.sidebar.success("서버에 연결되었습니다!")
        else:
            st.sidebar.error("서버 연결에 실패했습니다.")
    
    # 연결 상태 표시
    if st.session_state.socketio_client.connected:
        connection_status.success("✅ 서버 연결됨")
    else:
        connection_status.error("❌ 서버 미연결")
    
    # 판단 기준 가중치 설정
    st.sidebar.subheader("📊 판단 기준 설정")
    st.sidebar.write("각 항목의 가중치를 설정하세요 (합계: 1.0)")
    
    face_weight = st.sidebar.slider("얼굴 검출", 0.0, 1.0, 0.2, 0.05)
    recognition_weight = st.sidebar.slider("얼굴 인식", 0.0, 1.0, 0.3, 0.05)
    liveness_weight = st.sidebar.slider("생체 활동성", 0.0, 1.0, 0.15, 0.05)
    attention_weight = st.sidebar.slider("주의집중도", 0.0, 1.0, 0.25, 0.05)
    pose_weight = st.sidebar.slider("머리 자세", 0.0, 1.0, 0.1, 0.05)
    
    # 가중치 합계 확인
    total_weight = face_weight + recognition_weight + liveness_weight + attention_weight + pose_weight
    st.sidebar.write(f"**총 가중치: {total_weight:.2f}**")
    
    if abs(total_weight - 1.0) > 0.01:
        st.sidebar.warning("⚠️ 가중치 합계가 1.0이 아닙니다!")
    else:
        st.sidebar.success("✅ 가중치 설정 완료")
    
    # 가중치 딕셔너리 생성
    criteria_weights = {
        'face_detected': face_weight,
        'face_recognized': recognition_weight,
        'liveness': liveness_weight,
        'attention': attention_weight,
        'head_pose': pose_weight
    }
    
    # WebRTC 설정
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # 메인 컨텐츠 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 실시간 비디오 분석")
        
        # VideoProcessor 초기화/업데이트
        if st.session_state.video_processor is None:
            st.session_state.video_processor = VideoProcessor(criteria_weights)
        else:
            st.session_state.video_processor.update_criteria_weights(criteria_weights)
        
        # WebRTC 스트리머
        webrtc_ctx = webrtc_streamer(
            key="attendance-analyzer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: st.session_state.video_processor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )
    
    with col2:
        st.subheader("📊 실시간 분석 결과")
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        weights_placeholder = st.empty()
        
        # 가중치 표시
        with weights_placeholder.container():
            st.write("**현재 가중치 설정:**")
            for key, value in criteria_weights.items():
                st.write(f"- {key}: {value:.2f}")
        
        # 실시간 업데이트 루프
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            while True:
                try:
                    result = webrtc_ctx.video_processor.get_last_analysis()
                    
                    if result and result['timestamp'] > 0:
                        # 서버로 데이터 전송
                        if st.session_state.socketio_client.connected:
                            student_data = {
                                'id': student_id,
                                'name': student_name
                            }
                            st.session_state.socketio_client.send_attendance_data(
                                student_data, result
                            )
                        
                        # UI 업데이트
                        with status_placeholder.container():
                            status_color = {
                                "excellent": "🟢",
                                "present": "🟡",
                                "attention_needed": "🟠",
                                "absent": "🔴",
                                "error": "⚫"
                            }
                            
                            st.metric(
                                "출석 상태",
                                f"{status_color.get(result.get('attendance_status', 'error'), '⚪')} {result.get('attendance_status', 'ERROR').upper()}",
                                f"점수: {result.get('attendance_score', 0.0):.2f}"
                            )
                        
                        with metrics_placeholder.container():
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("얼굴 검출", "✅" if result.get('face_detected') else "❌")
                                st.metric("생체 활동성", "✅" if result.get('is_alive') else "❌")
                            with col_b:
                                st.metric("얼굴 인식", "✅" if result.get('face_recognized') else "❌")
                                st.metric("주의집중", "✅" if result.get('attention_focused') else "❌")
                            
                            st.subheader("머리 자세")
                            pose = result.get('head_pose', {'yaw':0, 'pitch':0, 'roll':0})
                            st.write(f"**Yaw:** {pose.get('yaw',0):.1f}°")
                            st.write(f"**Pitch:** {pose.get('pitch',0):.1f}°")
                            st.write(f"**Roll:** {pose.get('roll',0):.1f}°")
                            
                            st.metric("눈 깜박임", result.get('blink_count', 0))
                            
                            # 점수 구성 요소 표시
                            if 'score_components' in result:
                                st.subheader("점수 구성")
                                for component, score in result['score_components'].items():
                                    st.write(f"- {component}: {score:.3f}")
                    else:
                        with status_placeholder.container():
                            st.info("분석 결과를 기다리는 중...")
                    
                    time.sleep(0.2)
                except Exception as e:
                    logger.error(f"결과 표시 중 오류: {e}")
                    with status_placeholder.container():
                        st.error(f"분석 중 오류 발생: {str(e)}")
                    break
                
                if not webrtc_ctx.state.playing:
                    break
        else:
            with status_placeholder.container():
                st.info("웹캠을 시작하려면 'START' 버튼을 클릭하세요")
    
    # 하단 정보 표시
    st.markdown("---")
    with st.expander("📋 시스템 정보"):
        st.write("**분석 항목:**")
        st.write("- 얼굴 검출: 카메라에서 얼굴이 정상적으로 검출되는지 확인")
        st.write("- 얼굴 인식: 등록된 학생의 얼굴과 일치하는지 확인")
        st.write("- 생체 활동성: 눈 깜박임 등을 통한 실제 사람 여부 확인")
        st.write("- 주의집중도: 화면을 바라보고 있는지 집중도 측정")
        st.write("- 머리 자세: 올바른 수강 자세 유지 여부")
        
        st.write("**출석 등급:**")
        st.write("- 🟢 우수 (0.8 이상): 매우 집중적 수강")
        st.write("- 🟡 정상 (0.7 이상): 양호한 참여도")
        st.write("- 🟠 주의 (0.5 이상): 산만하거나 부분 참여")
        st.write("- 🔴 비정상 (0.5 미만): 출석 불인정")
        
        st.write("**서버 통신:**")
        st.write("- SocketIO를 통한 실시간 서버 통신")
        st.write("- 분석 결과 자동 전송 및 기록")
        st.write("- 네트워크 상태 모니터링")
        
        st.write("**교수자 설정:**")
        st.write("- 판단 기준별 가중치 실시간 조정 가능")
        st.write("- 강의 특성에 맞는 맞춤형 출석 기준 설정")
        st.write("- 설정 변경 시 즉시 반영")

if __name__ == "__main__":
    main()
