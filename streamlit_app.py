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
    def __init__(self, config=None):
        self.config = config  # config 참조 저장
        try:
            # MediaPipe 초기화 (한번만 초기화 - 메모리 누수 방지)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
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
        try:
            eye = np.array([(landmarks[point].x, landmarks[point].y) for point in eye_points])
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            ear = (A + B) / (2.0 * C) if C != 0 else 0.3
            logger.info(f"EAR 값: {ear:.3f}")  # 디버깅 로그 추가
            return ear
        except Exception as e:
            logger.error(f"EAR 계산 오류: {e}")
            return 0.3

    def detect_blink(self, landmarks):
        # 최신 MediaPipe 기준 눈 좌표 인덱스 사용
        left_eye_points = [33, 160, 158, 133, 153, 144]
        right_eye_points = [362, 385, 387, 263, 373, 380]
        left_ear = self.calculate_ear(landmarks, left_eye_points)
        right_ear = self.calculate_ear(landmarks, right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        logger.info(f"평균 EAR: {avg_ear:.3f}")  # 디버깅 로그 추가
        blink_threshold = 0.27  # 임계값 낮춤
        current_time = time.time()
        # 쿨타임 0.3초 적용
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
                logger.info(f"Head Pose: yaw={self.head_pose_angles['yaw']:.1f}, pitch={self.head_pose_angles['pitch']:.1f}, roll={self.head_pose_angles['roll']:.1f}")
            else:
                logger.warning("solvePnP 실패")
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
                    current_time = time.time()
                    if self.detect_blink(face_landmarks.landmark):
                        with self.analysis_lock:
                            self.blink_count += 1
                            self.last_blink_time = current_time
                        logger.debug(f"눈 깜박임 감지 (총 {self.blink_count}회)")
                    # 최근 2초 이내 깜박임이 있으면 is_alive True
                    with self.analysis_lock:
                        is_alive = (current_time - self.last_blink_time) < 2.0

                    # 머리 자세 추정
                    self.estimate_head_pose(face_landmarks.landmark, frame.shape)

                    # 주의집중도 판단 (roll 각도 보정, pitch 허용범위 완화)
                    with self.analysis_lock:
                        yaw_ok = abs(self.head_pose_angles['yaw']) < 30
                        pitch_ok = abs(self.head_pose_angles['pitch']) < 40
                        roll_val = self.head_pose_angles['roll']
                        roll_ok = abs(roll_val) < 25 or abs(abs(roll_val) - 180) < 25
                        attention_focused = yaw_ok and pitch_ok and roll_ok
                    if attention_focused:
                        logger.debug("주의집중 상태 양호")
                    else:
                        logger.debug(f"주의집중 저하: yaw={self.head_pose_angles['yaw']:.1f}, pitch={self.head_pose_angles['pitch']:.1f}, roll={self.head_pose_angles['roll']:.1f}")

            # 가중치/임계값 실시간 적용
            cfg = self.config if self.config else {
                'face_detected_weight': 0.2,
                'face_recognized_weight': 0.3,
                'liveness_weight': 0.15,
                'attention_weight': 0.25,
                'head_pose_weight': 0.1,
                'excellent_threshold': 0.8,
                'present_threshold': 0.7,
                'attention_needed_threshold': 0.5
            }
            score_components = {
                'face_detected': cfg['face_detected_weight'] if face_detected else 0.0,
                'face_recognized': cfg['face_recognized_weight'] if face_detected else 0.0,
                'liveness': cfg['liveness_weight'] if is_alive else 0.0,
                'attention': cfg['attention_weight'] if attention_focused else 0.0,
                'head_pose': cfg['head_pose_weight'] if attention_focused else 0.0
            }
            total_score = sum(score_components.values())

            # 출석 상태 결정
            if total_score >= cfg['excellent_threshold']:
                status = "excellent"
            elif total_score >= cfg['present_threshold']:
                status = "present"
            elif total_score >= cfg['attention_needed_threshold']:
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
    def __init__(self, config=None):
        self.frame_count = 0
        self.analysis_interval = 5  # 5프레임마다 분석 (성능과 반응성 균형)
        self.display_interval = 15  # 15프레임마다 화면 표시 업데이트
        self.display_result = None  # 화면 표시용 결과 저장
        self.config = config  # config 참조 저장
        
        # 분석기 초기화 (한번만)
        try:
            self.analyzer = AttendanceAnalyzer(config=self.config)
            logger.info("VideoProcessor 초기화 완료")
        except Exception as e:
            logger.error(f"VideoProcessor 초기화 실패: {e}")
            raise e

        # 분석 결과 저장용 변수
        self.last_analysis = None
        self.analysis_lock = Lock()

    def draw_analysis_result(self, frame):
        """분석 결과 시각화"""
        try:
            with self.analysis_lock:
                if not self.last_analysis:
                    return

                # 화면 표시 주기에 따라 결과 업데이트
                if self.frame_count % self.display_interval == 0:
                    self.display_result = self.last_analysis.copy()

                if not self.display_result:
                    return

                # 반투명한 배경 추가
                overlay = frame.copy()
                bg_height = 120  # 배경 높이
                cv2.rectangle(overlay, (5, 5), (400, bg_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                # 상태에 따른 색상 설정
                status_color = {
                    "excellent": (0, 255, 0),    # 녹색
                    "present": (0, 255, 255),    # 노란색
                    "attention_needed": (0, 165, 255),  # 주황색
                    "absent": (0, 0, 255),       # 빨간색
                    "error": (128, 128, 128)     # 회색
                }

                # 상태 표시
                color = status_color.get(self.display_result['attendance_status'], (128, 128, 128))
                
                # 텍스트에 검은색 외곽선 추가
                def putTextWithOutline(img, text, pos, font, scale, color, thickness=2):
                    # 외곽선 (검은색)
                    cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness*3)
                    # 내부 텍스트
                    cv2.putText(img, text, pos, font, scale, color, thickness)

                # 정보 표시 (외곽선 있는 텍스트)
                putTextWithOutline(frame, f"Status: {self.display_result['attendance_status'].upper()}", 
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
                putTextWithOutline(frame, f"Score: {self.display_result['attendance_score']:.2f}", 
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
                putTextWithOutline(frame, f"Blinks: {self.display_result['blink_count']}", 
                        (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        except Exception as e:
            logger.error(f"결과 시각화 중 오류: {e}")

    def recv(self, frame):
        """프레임 수신 및 처리"""
        try:
            # 프레임 카운트 증가
            self.frame_count += 1

            # 분석 간격마다 처리
            if self.frame_count % self.analysis_interval == 0:
                img = frame.to_ndarray(format="bgr24")
                
                # 프레임 분석
                result = self.analyzer.analyze_frame(img)
                
                # 분석 결과 저장 (스레드 안전)
                with self.analysis_lock:
                    self.last_analysis = result

                # 분석 결과 시각화
                self.draw_analysis_result(img)
                
                # 처리된 프레임 반환
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # 분석하지 않는 프레임도 시각화는 수행
            img = frame.to_ndarray(format="bgr24")
            self.draw_analysis_result(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            logger.error(f"프레임 처리 중 오류: {e}")
            return frame

    def get_last_analysis(self):
        """마지막 분석 결과 반환 (스레드 안전)"""
        with self.analysis_lock:
            if self.last_analysis:
                return self.last_analysis.copy()
            else:
                # 기본값 반환
                return {
                    'face_detected': False,
                    'face_recognized': False,
                    'is_alive': False,
                    'attention_focused': False,
                    'head_pose': {'yaw': 0, 'pitch': 0, 'roll': 0},
                    'blink_count': 0,
                    'attendance_score': 0.0,
                    'attendance_status': "error",
                    'timestamp': time.time()
                }

def render_customization_panel(role, config):
    st.markdown("---")
    st.subheader("분석 기준 커스터마이징")
    if role == "교수":
        st.info("분석 기준을 조정할 수 있습니다. (교수 전용)")
        config['face_detected_weight'] = st.slider("얼굴 검출 가중치", 0.0, 1.0, config.get('face_detected_weight', 0.2), 0.05)
        config['face_recognized_weight'] = st.slider("얼굴 인식 가중치", 0.0, 1.0, config.get('face_recognized_weight', 0.3), 0.05)
        config['liveness_weight'] = st.slider("생체 활동성 가중치", 0.0, 1.0, config.get('liveness_weight', 0.15), 0.05)
        config['attention_weight'] = st.slider("주의집중 가중치", 0.0, 1.0, config.get('attention_weight', 0.25), 0.05)
        config['head_pose_weight'] = st.slider("머리 자세 가중치", 0.0, 1.0, config.get('head_pose_weight', 0.1), 0.05)
        config['excellent_threshold'] = st.slider("'최우수' 임계값", 0.0, 1.0, config.get('excellent_threshold', 0.8), 0.01)
        config['present_threshold'] = st.slider("'출석' 임계값", 0.0, 1.0, config.get('present_threshold', 0.7), 0.01)
        config['attention_needed_threshold'] = st.slider("'주의필요' 임계값", 0.0, 1.0, config.get('attention_needed_threshold', 0.5), 0.01)
    else:
        st.info("설정값은 교수만 변경할 수 있습니다.")
        st.write(f"얼굴 검출 가중치: {config.get('face_detected_weight', 0.2)}")
        st.write(f"얼굴 인식 가중치: {config.get('face_recognized_weight', 0.3)}")
        st.write(f"생체 활동성 가중치: {config.get('liveness_weight', 0.15)}")
        st.write(f"주의집중 가중치: {config.get('attention_weight', 0.25)}")
        st.write(f"머리 자세 가중치: {config.get('head_pose_weight', 0.1)}")
        st.write(f"'최우수' 임계값: {config.get('excellent_threshold', 0.8)}")
        st.write(f"'출석' 임계값: {config.get('present_threshold', 0.7)}")
        st.write(f"'주의필요' 임계값: {config.get('attention_needed_threshold', 0.5)}")

# 메인 애플리케이션
def main():
    st.title("실시간 온라인 강의 출석 시스템")
    st.markdown("""
    이 시스템은 실시간으로 학생의 출석 상태를 모니터링합니다.
    - 👤 **얼굴 검출**: 학생의 얼굴이 화면에 있는지 확인
    - 👁️ **생체 활동성**: 눈 깜박임을 통한 실제 사람 여부 확인
    - 🎯 **주의집중도**: 머리 자세를 통한 수업 집중도 측정
    """)

    # 역할 선택
    role = st.sidebar.selectbox("역할을 선택하세요", ["교수", "학생"])

    # 분석 파라미터 config (세션 상태로 관리)
    if 'analyze_config' not in st.session_state:
        st.session_state['analyze_config'] = {
            'face_detected_weight': 0.2,
            'face_recognized_weight': 0.3,
            'liveness_weight': 0.15,
            'attention_weight': 0.25,
            'head_pose_weight': 0.1,
            'excellent_threshold': 0.8,
            'present_threshold': 0.7,
            'attention_needed_threshold': 0.5
        }
    config = st.session_state['analyze_config']

    # 커스터마이징 패널
    render_customization_panel(role, config)

    # STUN 서버 설정
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # 레이아웃
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📹 웹캠 화면")
        webrtc_ctx = webrtc_streamer(
            key="attendance-analyzer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: VideoProcessor(config=config),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,  # 동기 처리로 변경
        )

    with col2:
        st.subheader("📊 실시간 분석 결과")

        status_placeholder = st.empty()
        metrics_placeholder = st.empty()

        # 실시간 업데이트 루프
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            while True:
                try:
                    result = webrtc_ctx.video_processor.get_last_analysis()
                    if result:
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

if __name__ == "__main__":
    main()