// 출석 관리 시스템 서버 코드
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const { MongoClient } = require('mongodb');

// 익스프레스 앱 생성
const app = express();
app.use(cors());
app.use(express.json());
s
// HTTP 서버 생성
const server = http.createServer(app);

// Socket.IO 서버 생성
const io = socketIo(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// MongoDB 연결 설정
const mongoUrl = process.env.MONGO_URL || 'mongodb://localhost:27017';
const dbName = 'attendanceSystem';
let db;

// 데이터베이스 연결
async function connectToDatabase() {
    try {
        const client = new MongoClient(mongoUrl);
        await client.connect();
        console.log('MongoDB에 연결되었습니다');
        db = client.db(dbName);
        
        // 컬렉션 초기화
        await db.collection('students').createIndex({ studentId: 1 }, { unique: true });
        await db.collection('attendance').createIndex({ studentId: 1, timestamp: 1 });
        await db.collection('classrooms').createIndex({ roomId: 1 }, { unique: true });
        
        console.log('데이터베이스 초기화 완료');
    } catch (error) {
        console.error('데이터베이스 연결 오류:', error);
    }
}

// 활성 세션 관리 (메모리 내 저장)
const activeSessions = new Map();

// API 라우트 설정
app.get('/api/classrooms', async (req, res) => {
    try {
        const classrooms = await db.collection('classrooms').find({}).toArray();
        res.json(classrooms);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/attendance/:studentId', async (req, res) => {
    try {
        const { studentId } = req.params;
        const attendance = await db.collection('attendance')
            .find({ studentId })
            .sort({ timestamp: -1 })
            .limit(100)
            .toArray();
        res.json(attendance);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/realtime-status', (req, res) => {
    // 서버 상태 및 통계 정보
    const status = {
        activeSessions: activeSessions.size,
        server: {
            uptime: process.uptime(),
            memory: process.memoryUsage(),
            cpu: process.cpuUsage()
        }
    };
    res.json(status);
});

// Socket.IO 이벤트 핸들러
io.on('connection', (socket) => {
    console.log(`클라이언트 연결됨: ${socket.id}`);
    
    // 강의실 입장
    socket.on('joinClassroom', async (data) => {
        try {
            const { studentId, studentName, roomId } = data;
            
            // 학생 정보 저장 또는 업데이트
            await db.collection('students').updateOne(
                { studentId },
                { 
                    $set: { 
                        studentName,
                        lastActive: new Date(),
                        socketId: socket.id,
                        online: true
                    } 
                },
                { upsert: true }
            );
            
            // 강의실 정보 업데이트
            await db.collection('classrooms').updateOne(
                { roomId },
                { 
                    $set: { lastActive: new Date() },
                    $addToSet: { activeStudents: studentId }
                },
                { upsert: true }
            );
            
            // 소켓을 강의실 룸에 추가
            socket.join(roomId);
            
            // 메모리 내 세션 정보 업데이트
            activeSessions.set(socket.id, { studentId, roomId, lastUpdate: Date.now() });
            
            console.log(`학생 ${studentId}가 강의실 ${roomId}에 입장했습니다`);
            
            // 강의실 학생들에게 새 학생 입장 알림
            socket.to(roomId).emit('studentJoined', { studentId, studentName });
            
            // 연결된 클라이언트에 입장 확인 응답
            socket.emit('joinedClassroom', { 
                success: true, 
                message: `강의실 ${roomId}에 성공적으로 입장했습니다` 
            });
        } catch (error) {
            console.error('강의실 입장 오류:', error);
            socket.emit('joinedClassroom', { 
                success: false, 
                message: `오류: ${error.message}` 
            });
        }
    });
    
    // 실시간 출석 정보 업데이트
    socket.on('realtimeAttendanceUpdate', async (data) => {
        try {
            const { attendanceData } = data;
            const sessionInfo = activeSessions.get(socket.id);
            
            if (!sessionInfo) {
                socket.emit('error', { message: '활성 세션을 찾을 수 없습니다. 다시 로그인하세요.' });
                return;
            }
            
            const { studentId, roomId } = sessionInfo;
            
            // 세션 활성 시간 업데이트
            sessionInfo.lastUpdate = Date.now();
            activeSessions.set(socket.id, sessionInfo);
            
            // 출석 데이터에 필요한 정보 추가
            const enrichedData = {
                ...attendanceData,
                studentId,
                roomId,
                timestamp: new Date(),
                createdAt: new Date()
            };
            
            // 출석 정보를 데이터베이스에 저장
            await db.collection('attendance').insertOne(enrichedData);
            
            // 학생 상태 업데이트
            await db.collection('students').updateOne(
                { studentId },
                { 
                    $set: { 
                        lastActive: new Date(),
                        lastStatus: attendanceData.attendance_status,
                        lastScore: attendanceData.attendance_score
                    } 
                }
            );
            
            // 교수자에게 업데이트된 정보 전송 (instructor_ 접두사가 있는 소켓 ID로)
            const instructors = Array.from(io.sockets.adapter.rooms.get(roomId) || [])
                .filter(id => id.startsWith('instructor_'));
                
            if (instructors.length > 0) {
                io.to(instructors).emit('studentStatusUpdate', {
                    studentId,
                    status: attendanceData.attendance_status,
                    score: attendanceData.attendance_score,
                    timestamp: new Date()
                });
            }
            
            // 클라이언트에 처리 완료 응답
            socket.emit('attendanceUpdated', { 
                success: true, 
                timestamp: new Date() 
            });
        } catch (error) {
            console.error('출석 정보 업데이트 오류:', error);
            socket.emit('attendanceUpdated', { 
                success: false, 
                message: `오류: ${error.message}` 
            });
        }
    });
    
    // 연결 종료
    socket.on('disconnect', async () => {
        try {
            const sessionInfo = activeSessions.get(socket.id);
            
            if (sessionInfo) {
                const { studentId, roomId } = sessionInfo;
                
                // 학생 상태 업데이트
                await db.collection('students').updateOne(
                    { studentId },
                    { $set: { online: false, lastDisconnect: new Date() } }
                );
                
                // 강의실에서 학생 제거
                await db.collection('classrooms').updateOne(
                    { roomId },
                    { $pull: { activeStudents: studentId } }
                );
                
                // 강의실 학생들에게 퇴장 알림
                socket.to(roomId).emit('studentLeft', { studentId });
                
                // 활성 세션에서 제거
                activeSessions.delete(socket.id);
                
                console.log(`학생 ${studentId}가 강의실 ${roomId}에서 퇴장했습니다`);
            }
        } catch (error) {
            console.error('연결 종료 처리 오류:', error);
        }
    });
});

// 비활성 세션 정리 (5분마다 실행)
setInterval(async () => {
    const now = Date.now();
    const inactiveTimeout = 5 * 60 * 1000; // 5분
    
    for (const [socketId, session] of activeSessions.entries()) {
        if (now - session.lastUpdate > inactiveTimeout) {
            try {
                const { studentId, roomId } = session;
                
                // 학생 상태 업데이트
                await db.collection('students').updateOne(
                    { studentId },
                    { $set: { online: false, lastDisconnect: new Date() } }
                );
                
                // 강의실에서 학생 제거
                await db.collection('classrooms').updateOne(
                    { roomId },
                    { $pull: { activeStudents: studentId } }
                );
                
                // 활성 세션에서 제거
                activeSessions.delete(socketId);
                
                console.log(`비활성 세션 제거: 학생 ${studentId}, 소켓 ${socketId}`);
            } catch (error) {
                console.error('비활성 세션 정리 오류:', error);
            }
        }
    }
}, 5 * 60 * 1000);

// 서버 시작
const PORT = process.env.PORT || 3000;
server.listen(PORT, async () => {
    console.log(`서버가 포트 ${PORT}에서 실행 중입니다`);
    await connectToDatabase();
});
