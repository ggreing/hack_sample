import asyncio
import socketio
import cv2
import dlib
import numpy as np
import mediapipe as mp
import face_recognition
import time
import json
from threading import Thread
import queue

# 서버 설정
SERVER_URL = "http://localhost:3000"

class RealTimeAttendanceClient:
    def __init__(self, student_id, student_name):
        # Socket.IO 클라이언트 초기화
        self.sio = socketio.AsyncClient(
            ssl_verify=False,
            engineio_logger=False
        )

        # 학생 정보
        self.student_id = student_id
        self.student_name = student_name
        self.room_id = f"class_{student_id}"

        # MediaPipe 초기화 (더 간단한 설정)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 카메라 설정
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 분석 변수 초기화
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.head_pose_angles = {'yaw': 0, 'pitch': 0, 'roll': 0}

        # 실시간 처리를 위한 큐
        self.frame_queue = queue.Queue(maxsize=5)
        self.analysis_running = True

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

        # 출석 상태 변수
        self.attendance_status = "not_detected"
        self.last_update_time = time.time()

        # Socket.IO 이벤트 핸들러 등록
        self.setup_socket_events()

    def setup_socket_events(self):
        """Socket.IO 이벤트 핸들러 설정"""
        @self.sio.event
        async def connect():
            print(f"🔗 서버에 연결됨 - 학생: {self.student_name}")
            await self.join_classroom()

        @self.sio.event
        async def disconnect():
            print("❌ 서버 연결 해제")

        @self.sio.on("attendanceUpdate")
        async def on_attendance_update(data):
            print(f"📊 출석 상태 업데이트: {data}")

        @self.sio.on("classroomJoined")
        async def on_classroom_joined(data):
            print(f"🎓 강의실 입장 완료: {data}")

        @self.sio.on("liveAnalysisResult")
        async def on_live_analysis(data):
            # 다른 학생들의 분석 결과는 간단히 로그만
            pass

    async def join_classroom(self):
        """강의실 입장"""
        join_data = {
            'student_id': self.student_id,
            'student_name': self.student_name,
            'room_id': self.room_id,
            'timestamp': time.time()
        }
        await self.sio.emit('joinClassroom', join_data)

    def calculate_ear(self, landmarks, eye_points):
        """Eye Aspect Ratio 계산 (눈 깜박임 감지)"""
        if len(eye_points) < 6:
            return 0.3  # 기본값

        try:
            eye = np.array([(landmarks[point].x, landmarks[point].y) for point in eye_points[:6]])
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            ear = (A + B) / (2.0 * C)
            return ear
        except:
            return 0.3

    def detect_blink(self, landmarks):
        """눈 깜박임 감지"""
        # MediaPipe 얼굴 랜드마크 인덱스 (간소화)
        left_eye_points = [33, 7, 163, 144, 145, 153]
        right_eye_points = [362, 382, 381, 380, 374, 373]

        left_ear = self.calculate_ear(landmarks, left_eye_points)
        right_ear = self.calculate_ear(landmarks, right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0

        return avg_ear < 0.25

    def estimate_head_pose(self, landmarks, frame_shape):
        """머리 자세 추정 (간소화)"""
        try:
            # 2D 이미지 포인트 추출
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

                if sy > 1e-6:
                    yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
                    pitch = np.arctan2(-rotation_matrix[2,0], sy)
                    roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                else:
                    yaw = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                    pitch = np.arctan2(-rotation_matrix[2,0], sy)
                    roll = 0

                self.head_pose_angles = {
                    'yaw': np.degrees(yaw),
                    'pitch': np.degrees(pitch),
                    'roll': np.degrees(roll)
                }
        except Exception as e:
            # 에러 발생 시 기본값 유지
            pass

    def analyze_frame(self, frame):
        """프레임 실시간 분석"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        face_detected = False
        face_recognized = False
        is_alive = False
        attention_focused = False

        if results.multi_face_landmarks:
            face_detected = True
            face_recognized = True  # 간단히 얼굴 검출로 대체

            for face_landmarks in results.multi_face_landmarks:
                # 눈 깜박임 감지
                if self.detect_blink(face_landmarks.landmark):
                    current_time = time.time()
                    if current_time - self.last_blink_time > 0.3:
                        self.blink_count += 1
                        self.last_blink_time = current_time
                        is_alive = True

                # 머리 자세 추정
                self.estimate_head_pose(face_landmarks.landmark, frame.shape)

                # 주의집중도 판단 (머리 자세 기준)
                yaw_ok = abs(self.head_pose_angles['yaw']) < 30
                pitch_ok = abs(self.head_pose_angles['pitch']) < 20
                roll_ok = abs(self.head_pose_angles['roll']) < 25
                attention_focused = yaw_ok and pitch_ok and roll_ok

        # 종합 참여도 점수 계산
        score_components = {
            'face_detected': 0.2 if face_detected else 0.0,
            'face_recognized': 0.3 if face_recognized else 0.0,
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

        return {
            'face_detected': face_detected,
            'face_recognized': face_recognized,
            'is_alive': is_alive,
            'attention_focused': attention_focused,
            'head_pose': self.head_pose_angles,
            'blink_count': self.blink_count,
            'attendance_score': total_score,
            'attendance_status': status,
            'timestamp': time.time()
        }

    async def send_realtime_update(self, analysis_result):
        """실시간 출석 상태 업데이트 전송"""
        update_data = {
            'student_id': self.student_id,
            'room_id': self.room_id,
            'analysis': analysis_result,
            'session_time': time.time() - self.last_update_time
        }

        try:
            await self.sio.emit('realtimeAttendanceUpdate', update_data)
        except Exception as e:
            print(f"❌ 실시간 업데이트 전송 실패: {e}")

    def frame_capture_thread(self):
        """별도 스레드에서 프레임 캡처"""
        while self.analysis_running:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

            time.sleep(0.033)  # 약 30 FPS

    async def realtime_analysis_loop(self):
        """실시간 분석 메인 루프"""
        # 프레임 캡처 스레드 시작
        capture_thread = Thread(target=self.frame_capture_thread)
        capture_thread.daemon = True
        capture_thread.start()

        print("🎥 실시간 출석 분석 시작...")

        while self.analysis_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()

                    # 프레임 분석
                    analysis_result = self.analyze_frame(frame)

                    # 실시간 업데이트 전송 (매 프레임마다)
                    await self.send_realtime_update(analysis_result)

                    # 화면에 분석 결과 표시
                    self.display_analysis_result(frame, analysis_result)

                    # 프레임 출력
                    cv2.imshow('Real-time Attendance Monitor', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                await asyncio.sleep(0.01)  # 10ms

            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"❌ 분석 오류: {e}")
                await asyncio.sleep(0.1)

    def display_analysis_result(self, frame, result):
        """분석 결과를 프레임에 표시"""
        status_colors = {
            "excellent": (0, 255, 0),
            "present": (0, 200, 255),
            "attention_needed": (0, 165, 255),
            "absent": (0, 0, 255)
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

        # 머리 자세 정보
        pose = result['head_pose']
        cv2.putText(frame, f"Yaw: {pose['yaw']:.1f}°", (10, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitch: {pose['pitch']:.1f}°", (10, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Roll: {pose['roll']:.1f}°", (10, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    async def connect_and_start(self):
        """서버 연결 및 실시간 분석 시작"""
        try:
            print(f"🔄 서버 연결 시도 중: {SERVER_URL}")
            await self.sio.connect(
                SERVER_URL,
                transports=['websocket'],
                headers={'Content-Type': 'application/json'}
            )

            await self.realtime_analysis_loop()

        except Exception as e:
            print(f"❌ 연결 또는 분석 오류: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """리소스 정리"""
        self.analysis_running = False
        self.cap.release()
        cv2.destroyAllWindows()
        if self.sio.connected:
            await self.sio.disconnect()
        print("🧹 리소스 정리 완료")

# 메인 실행 함수
async def main():
    # 학생 정보 설정
    STUDENT_ID = input("학생 ID를 입력하세요 (기본값: 2024001): ").strip() or "2024001"
    STUDENT_NAME = input("학생 이름을 입력하세요 (기본값: 김철수): ").strip() or "김철수"

    print(f"\n🎓 학생 정보:")
    print(f"   ID: {STUDENT_ID}")
    print(f"   이름: {STUDENT_NAME}")
    print(f"\n📋 사용 방법:")
    print("   - 웹캠 창에서 'q' 키를 누르면 종료됩니다.")
    print("   - 서버가 실행 중인지 확인하세요.")
    print("\n🚀 시스템 시작 중...\n")

    # 클라이언트 생성
    client = RealTimeAttendanceClient(STUDENT_ID, STUDENT_NAME)

    # 연결 및 실시간 분석 시작
    await client.connect_and_start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 프로그램 종료")
    except Exception as e:
        print(f"\n❌ 프로그램 오류: {e}")
