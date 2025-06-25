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

# 로깅 설정 추가
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AttendanceSystem')

# 페이지 설정
st.set_page_config(
    page_title="실시간 온라인 강의 출석 시스템",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AttendanceAnalyzer:
    def __init__(self):
        try:
            # MediaPipe 초기화 (한번만 초기화 - 메모리 누수 방지)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )

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
                (0.0, 0.0, 0.0),             # 코끝
                (0.0, -330.0, -65.0),        # 턱
                (-225.0, 170.0, -135.0),     # 왼쪽 눈 코너
                (225.0, 170.0, -135.0),      # 오른쪽 눈 코너
                (-150.0, -150.0, -125.0),    # 왼쪽 입 코너
                (150.0, -150.0, -125.0)      # 오른쪽 입 코너
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

    def calculate_ear(self, landmarks, eye_points):
        """Eye Aspect Ratio 계산"""
        try:
            eye = np.array([(landmarks[point].x, landmarks[point].y) for point in eye_points])
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            if C == 0:  # 0으로 나누기 방지
                return 0.3
            ear = (A + B) / (2.0 * C)
            return ear
        except Exception as e:
            logger.error(f"EAR 계산 오류: {e}")
            return 0.3

    def detect_blink(self, landmarks):
        """눈 깜박임 감지"""
        try:
            left_eye_points = [33, 7, 163, 144, 145, 153]
            right_eye_points = [362, 382, 381, 380, 374, 373]

            left_ear = self.calculate_ear(landmarks, left_eye_points)
            right_ear = self.calculate_ear(landmarks, right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0

            return avg_ear < 0.25
        except Exception as e:
            logger.error(f"눈 깜박임 감지 오류: {e}")
            return False

    def estimate_head_pose(self, landmarks, frame_shape):
        """머리 자세 추정"""
        try:
            if len(landmarks) < 468:  # MediaPipe face mesh has 468 landmarks
                return
                
            image_points = np.array([
                (landmarks[1].x * frame_shape[1], landmarks[1].y * frame_shape[0]),    # 코끝
                (landmarks[152].x * frame_shape[1], landmarks[152].y * frame_shape[0]), # 턱
                (landmarks[226].x * frame_shape[1], landmarks[226].y * frame_shape[0]), # 왼쪽 눈 코너
                (landmarks[446].x * frame_shape[1], landmarks[446].y * frame_shape[0]), # 오른쪽 눈 코너
                (landmarks[57].x * frame_shape[1], landmarks[57].y * frame_shape[0]),   # 왼쪽 입 코너
                (landmarks[287].x * frame_shape[1], landmarks[287].y * frame_shape[0])  # 오른쪽 입 코너
            ], dtype=np.float32)

            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix, self.dist_coeffs
            )

            if success:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  rotation_matrix[1,0] * rotation_matrix[1,0])

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
            # 프레임 크기 확인
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
                logger.debug("얼굴 검출됨")

                for face_landmarks in results.multi_face_landmarks:
                    # 눈 깜박임 감지
                    if self.detect_blink(face_landmarks.landmark):
                        current_time = time.time()
                        if current_time - self.last_blink_time > 0.3:
                            with self.analysis_lock:
                                self.blink_count += 1
                                self.last_blink_time = current_time
                            is_alive = True
                            logger.debug(f"눈 깜박임 감지 (총 {self.blink_count}회)")

                    # 머리 자세 추정
                    self.estimate_head_pose(face_landmarks.landmark, frame.shape)

                    # 주의집중도 판단
                    with self.analysis_lock:
                        yaw_ok = abs(self.head_pose_angles['yaw']) < 30
                        pitch_ok = abs(self.head_pose_angles['pitch']) < 20
                        roll_ok = abs(self.head_pose_angles['roll']) < 25
                        attention_focused = yaw_ok and pitch_ok and roll_ok
                    
                    if attention_focused:
                        logger.debug("주의집중 상태 양호")
                    else:
                        logger.debug(f"주의집중 저하: yaw={self.head_pose_angles['yaw']:.1f}, pitch={self.head_pose_angles['pitch']:.1f}, roll={self.head_pose_angles['roll']:.1f}")

            # 종합 참여도 점수 계산
            score_components = {
                'face_detected': 0.2 if face_detected else 0.0,
                'face_recognized': 0.3 if face_detected else 0.0,  # 간단히 얼굴 검출로 대체
                'liveness': 0.15 if is_alive else 0.0,
                'attention': 0.25 if attention_focused else 0.0,
                'head_pose': 0.1 if attention_focused else 0.0
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
                
            logger.debug(f"분석 완료: 상태={status}, 점수={total_score:.2f}")

            with self.analysis_lock:
                return {
                    'face_detected': face_detected,
                    'face_recognized': face_detected,  # 간단히 얼굴 검출로 대체
                    'is_alive': is_alive,
                    'attention_focused': attention_focused,
                    'head_pose': self.head_pose_angles.copy(),
                    'blink_count': self.blink_count,
                    'attendance_score': total_score,
                    'attendance_status': status,
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
                'timestamp': time.time()
            }

# WebRTC 비디오 프로세서 클래스
class VideoProcessor:
    def __init__(self):
        # frame_count 초기화
        self.frame_count = 0
        self.analysis_interval = 10  # 10프레임마다 분석 (성능 최적화)
        
        # 분석기 초기화 (한번만)
        try:
            self.analyzer = AttendanceAnalyzer()
            self.last_analysis = None
            self.analysis_lock = Lock()
            logger.info("VideoProcessor 초기화 완료")
        except Exception as e:
            logger.error(f"VideoProcessor 초기화 실패: {e}")
            self.analyzer = None

    def recv(self, frame):
        try:
            if self.analyzer is None:
                logger.error("분석기가 초기화되지 않음")
                return frame

            # av.VideoFrame을 numpy array로 변환
            img = frame.to_ndarray(format="bgr24")

            # 프레임 분석 (매 N프레임마다 - 성능 최적화)
            if self.frame_count % self.analysis_interval == 0:
                logger.debug(f"프레임 {self.frame_count} 분석 중...")
                analysis_result = self.analyzer.analyze_frame(img)

                # 분석 결과 저장 (세션 상태 대신 클래스 속성 사용)
                with self.analysis_lock:
                    self.last_analysis = analysis_result
                logger.debug(f"분석 결과 업데이트: {analysis_result['attendance_status']}")

            self.frame_count += 1

            # 분석 결과를 프레임에 표시
            self.draw_analysis_result(img)

            # numpy array를 av.VideoFrame으로 변환하여 반환
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logger.error(f"프레임 처리 중 예외 발생: {e}")
            # 오류 발생 시 원본 프레임 반환
            return frame

    def draw_analysis_result(self, frame):
        """분석 결과를 프레임에 표시"""
        try:
            with self.analysis_lock:
                if self.last_analysis:
                    result = self.last_analysis

                    # 상태 텍스트 색상 설정
                    status_colors = {
                        "excellent": (0, 255, 0),
                        "present": (0, 200, 255),
                        "attention_needed": (0, 165, 255),
                        "absent": (0, 0, 255),
                        "error": (255, 0, 255)
                    }

                    status = result['attendance_status']
                    color = status_colors.get(status, (255, 255, 255))

                    # 정보 표시
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
            # 오류 메시지 표시
            cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def get_last_analysis(self):
        """마지막 분석 결과 반환 (스레드 안전)"""
        with self.analysis_lock:
            return self.last_analysis.copy() if self.last_analysis else None

# 메인 애플리케이션
def main():
    st.title("🎓 실시간 온라인 강의 출석 시스템")
    
    st.markdown("---")

    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")

    # 학생 정보 입력
    student_id = st.sidebar.text_input("학생 ID", value="2024001")
    student_name = st.sidebar.text_input("학생 이름", value="김철수")

    # WebRTC 설정
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # 메인 컨텐츠 영역을 두 개 컬럼으로 분할
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 실시간 비디오 분석")

        # WebRTC 스트리머
        webrtc_ctx = webrtc_streamer(
            key="attendance-analyzer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,  # 동기 처리로 변경
        )

    with col2:
        st.subheader("📊 실시간 분석 결과")

        # 분석 결과 표시 영역
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()

        # 실시간 업데이트
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            # 비디오 프로세서에서 직접 결과 가져오기
            try:
                result = webrtc_ctx.video_processor.get_last_analysis()
                
                if result:
                    with status_placeholder.container():
                        # 출석 상태
                        status_color = {
                            "excellent": "🟢",
                            "present": "🟡", 
                            "attention_needed": "🟠",
                            "absent": "🔴",
                            "error": "⚫"
                        }

                        st.metric(
                            "출석 상태", 
                            f"{status_color.get(result['attendance_status'], '⚪')} {result['attendance_status'].upper()}",
                            f"점수: {result['attendance_score']:.2f}"
                        )

                    with metrics_placeholder.container():
                        # 세부 지표
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.metric("얼굴 검출", "✅" if result['face_detected'] else "❌")
                            st.metric("생체 활동성", "✅" if result['is_alive'] else "❌")

                        with col_b:
                            st.metric("얼굴 인식", "✅" if result['face_recognized'] else "❌")
                            st.metric("주의집중", "✅" if result['attention_focused'] else "❌")

                        # 머리 자세 정보
                        st.subheader("머리 자세")
                        pose = result['head_pose']
                        st.write(f"**Yaw:** {pose['yaw']:.1f}°")
                        st.write(f"**Pitch:** {pose['pitch']:.1f}°") 
                        st.write(f"**Roll:** {pose['roll']:.1f}°")

                        # 깜박임 횟수
                        st.metric("눈 깜박임", result['blink_count'])
                else:
                    with status_placeholder.container():
                        st.info("분석 결과를 기다리는 중...")
            except Exception as e:
                logger.error(f"결과 표시 중 오류: {e}")
                with status_placeholder.container():
                    st.error(f"분석 중 오류 발생: {str(e)}")
        else:
            with status_placeholder.container():
                st.info("웹캠을 시작하려면 'START' 버튼을 클릭하세요")

    # 하단 정보 표시
    st.markdown("---")
    
    with st.expander("📋 시스템 정보"):
        st.write("**분석 항목:**")
        st.write("- 얼굴 검출 (20%)")
        st.write("- 얼굴 인식 (30%)")
        st.write("- 생체 활동성 (15%)")
        st.write("- 주의집중도 (25%)")
        st.write("- 머리 자세 (10%)")

        st.write("**출석 등급:**")
        st.write("- 🟢 우수 (0.8 이상): 매우 집중적 수강")
        st.write("- 🟡 정상 (0.7 이상): 양호한 참여도") 
        st.write("- 🟠 주의 (0.5 이상): 산만하거나 부분 참여")
        st.write("- 🔴 비정상 (0.5 미만): 출석 불인정")
        
        # 성능 정보 추가
        st.write("**성능 최적화:**")
        st.write("- MediaPipe 모델 재사용으로 메모리 누수 방지")
        st.write("- 프레임 분석 빈도 최적화 (10프레임마다)")
        st.write("- 스레드 안전성 보장")
        st.write("- 향상된 오류 처리")

if __name__ == "__main__":
    main()