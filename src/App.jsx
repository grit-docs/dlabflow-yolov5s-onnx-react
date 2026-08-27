/**
 * D-Lab Flow 실시간 객체 탐지 데모
 *
 * 이 애플리케이션은 ONNX Runtime을 사용하여 YOLOv5 모델로 실시간 객체 탐지를 수행합니다.
 * 브라우저에서 완전히 동작하며 사용자의 카메라를 통해 객체를 탐지하고 바운딩 박스로 표시합니다.
 */
import { useState, useRef, useEffect, useCallback, memo } from 'react'
import Webcam from 'react-webcam'
import * as ort from 'onnxruntime-web'
import 'onnxruntime-web/webgl'; // 🚀 WebGL 백엔드를 명시적으로 활성화
import 'onnxruntime-web/webgpu'; // 🚀 WebGPU 백엔드 명시적 추가
import './App.css'

// 🚀 onnxruntime-web WASM 파일이 있는 폴더 경로를 지정 (라이브러리가 자동으로 필요한 파일을 찾도록 함)
ort.env.wasm.wasmPaths = '/ort-wasm-files/';

// confidenceThreshold는 이제 customThreshold state로 관리됩니다.

// 🎯 Icon 컴포넌트를 App 함수 밖으로 이동시켜 불필요한 재정의 방지
const Icon = memo(({ name, className }) => {
  const icons = {
    loader: <div className="icon-loader"></div>,
    camera: <div className="icon-camera"></div>,
    power: <div className="icon-power"></div>,
    zap: <div className="icon-zap"></div>,
    logo: <img src="/logo_icon.svg" alt="바운딩 박스" className="icon-logo" style={{ width: '50px', height: '50px' }} />,
    logo2: <img src="/logo_icon.svg" alt="바운딩 박스" className="icon-logo" />
  };
  return <div className={`icon ${className || ''}`}>{icons[name]}</div>;
});
Icon.displayName = 'Icon';

// 🎯 슬라이더의 잦은 업데이트로 인한 성능 저하를 막기 위한 최적화된 컴포넌트
const ConfidenceSlider = memo(({ initialThreshold, onThresholdChange, disabled }) => {
  const [value, setValue] = useState(initialThreshold);

  // 부모 컴포넌트에서 threshold가 변경될 경우(예: 초기화) 동기화
  useEffect(() => {
    setValue(initialThreshold);
  }, [initialThreshold]);

  // 슬라이더를 움직일 때 내부 값만 업데이트 (빠른 시각적 피드백)
  const handleChange = (e) => {
    setValue(Number(e.target.value) / 100);
  };

  // 슬라이더 조작이 끝났을 때만 부모의 상태를 업데이트 (무거운 로직 실행)
  const handleRelease = () => {
    onThresholdChange(value);
  };

  return (
    <div className="confidence-slider-container" style={{
      display: 'flex',
      flexDirection: 'column', // 세로로 배치
      alignItems: 'stretch',   // 자식 요소를 꽉 채움
      width: '100%',           // 부모 너비에 맞춤
    }}>
      <label htmlFor="confidence-slider" style={{
        fontSize: '0.85rem',
        whiteSpace: 'nowrap',
        textAlign: 'right',     // 오른쪽 정렬
        marginBottom: '0.25rem' // 슬라이더와 간격
      }}>
        신뢰도: {Math.round(value * 100)}%
      </label>
      <input
        id="confidence-slider"
        type="range"
        min="1"
        max="100"
        value={value * 100}
        onChange={handleChange}
        onMouseUp={handleRelease}
        onTouchEnd={handleRelease}
        className="confidence-slider"
        disabled={disabled}
        style={{
          width: '100%' // 컨테이너 너비에 맞춤
        }}
      />
    </div>
  );
});
ConfidenceSlider.displayName = 'ConfidenceSlider'; // 디버깅을 위한 이름 설정

// 🚀 UI 컴포넌트들을 App 함수 밖으로 이동시키고 memo로 감싸 불필요한 재정의 방지
const Card = memo(({ className, children }) => (
    <div className={`card ${className || ''}`}>
      {children}
    </div>
));
Card.displayName = 'Card';

const Button = memo(({ onClick, disabled, className, children }) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className={`button ${className || ''}`}
    >
      {children}
    </button>
));
Button.displayName = 'Button';

const Badge = memo(({ className, children }) => (
    <span className={`badge ${className || ''}`}>
    {children}
  </span>
));
Badge.displayName = 'Badge';

// 🚀 시작 카드를 별도 컴포넌트로 분리하고 memo로 감싸 불필요한 재렌더링 방지
const StartCard = memo(({ onStart, isModelLoaded }) => {
  return (
    <Card className="start-card">
      <div>
        <Icon name="logo2" />
      </div>
      <h2>D-Lab Flow 실시간 객체 탐지 데모</h2>
      <p>카메라 시작 버튼을 눌러주세요</p>
      <Button
          onClick={onStart}
          disabled={!isModelLoaded}
          className="start-button"
      >
        <Icon name="camera" />
        카메라 시작
      </Button>
    </Card>
  );
});
StartCard.displayName = 'StartCard';

/**
 * ONNX 모델 파일의 메타데이터에서 클래스 이름을 추출합니다.
 * 이 함수는 onnxruntime-web에 메타데이터 읽기 API가 없어 바이너리 파일을 직접 파싱하는 휴리스틱을 사용합니다.
 * 모델의 'names' 메타데이터가 특정 문자열 형식으로 저장되어 있다고 가정합니다.
 * @param {string} modelPath - ONNX 모델 파일의 경로
 * @returns {Promise<string[]>} - 클래스 이름 배열
 */
async function extractClassesFromModel(modelPath) {
  try {
    const response = await fetch(modelPath);
    if (!response.ok) return [];

    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    const decoder = new TextDecoder('utf-8');

    // "names" 라는 키워드의 바이트 시퀀스를 찾습니다.
    const keyToSearch = "names";
    const keyBytes = keyToSearch.split('').map(c => c.charCodeAt(0));

    // 파일 전체에서 "names" 문자열의 위치를 찾습니다.
    for (let i = 0; i < bytes.length - keyBytes.length; i++) {
      let found = true;
      for (let j = 0; j < keyBytes.length; j++) {
        if (bytes[i + j] !== keyBytes[j]) {
          found = false;
          break;
        }
      }

      if (found) {
        // "names"를 찾았다면, 그 주변에서 '{...}' 형태의 문자열을 찾습니다.
        // 이는 protobuf 구조에 대한 매우 단순화된 가정입니다.
        for (let k = i + keyBytes.length; k < bytes.length - 10; k++) {
          if (bytes[k] === '{'.charCodeAt(0)) { // '{'
            let braceCount = 1;
            let endBracketPos = -1;
            
            // 짝이 맞는 '}'를 찾습니다.
            for (let l = k + 1; l < bytes.length; l++) {
              if (bytes[l] === '{'.charCodeAt(0)) braceCount++;
              if (bytes[l] === '}'.charCodeAt(0)) braceCount--;
              if (braceCount === 0) {
                endBracketPos = l;
                break;
              }
            }

            if (endBracketPos !== -1) {
              const potentialStringBytes = bytes.subarray(k, endBracketPos + 1);
              const namesString = decoder.decode(potentialStringBytes);
              
              // 추출된 문자열이 클래스 정보 형식과 맞는지 확인합니다.
              if (namesString.includes(":") && namesString.includes("'")) {
                const classRegex = /(\d+):\s*'([^']+)'/g;
                const classMap = {};
                let classMatch;
                while ((classMatch = classRegex.exec(namesString)) !== null) {
                  classMap[parseInt(classMatch[1], 10)] = classMatch[2];
                }
                
                const classKeys = Object.keys(classMap).map(Number).sort((a,b) => a-b);
                const classArray = classKeys.map(key => classMap[key]);

                if (classArray.length > 0) {
                  console.log(`✅ 모델 메타데이터에서 ${classArray.length}개의 클래스 추출 성공.`);
                  return classArray;
                }
              }
            }
          }
        }
      }
    }
  } catch (e) {
    console.error("모델 메타데이터에서 클래스 추출 중 오류 발생:", e);
  }
  return []; // 실패 시 빈 배열 반환
}

function App() {
  // Refs
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const sessionRef = useRef(null);
  const animationRef = useRef(null);
  const streamRef = useRef(null);
  const scaleCanvasRef = useRef(document.createElement('canvas'));
  // 📱 성능 최적화를 위한 임시 canvas 재사용
  const tempCanvasRef = useRef(document.createElement('canvas'));
  const sourceCanvasRef = useRef(document.createElement('canvas'));
  const modelLoadingStartedRef = useRef(false); // 🚀 모델 로딩 이중 실행 방지 플래그

  const detectionMemoryRef = useRef([]); // 최신 탐지 결과 + 유지할 이전 결과 포함
  const MAX_MISSED_FRAMES = 3; // 객체가 안 보여도 몇 프레임 유지

  // State variables
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [classes, setClasses] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [detections, setDetections] = useState([]);
  const [currentFPS, setCurrentFPS] = useState(0);
  const [debugMode] = useState(false); // 디버깅 모드 활성화 (문제 진단용)
  const [customThreshold, setCustomThreshold] = useState(0.5); // Custom threshold 조절
  const thresholdRef = useRef(0.5); // 🔧 실시간 threshold를 위한 ref
  
  // 🎯 스마트 카메라 반전 시스템 State
  const [cameraFacing, setCameraFacing] = useState("user"); // PC 기본값: "user" (웹캠)
  const [shouldFlipCamera, setShouldFlipCamera] = useState(true); // PC 기본값: 미러링
  const [isMobile, setIsMobile] = useState(false); // 모바일 감지
  const [deviceType, setDeviceType] = useState(null); // "desktop" or "mobile" - null로 시작하여 경쟁 조건 방지
  const [manualFlip, setManualFlip] = useState(null); // 수동 반전 설정 (null = 자동)
  const [isSwitchingCamera, setIsSwitchingCamera] = useState(false); // 카메라 전환 중 상태
  const [webcamKey, setWebcamKey] = useState(Date.now()); // 📸 Webcam 강제 리마운트를 위한 key

  // 🎯 현재 활성화된 가속 엔진 백엔드 표시용 State 추가
  const [activeBackend, setActiveBackend] = useState('확인 중...');
  // auto: WebGPU → WebGL → CPU(WASM) 순서로 자동 선택
  const [selectedBackend, setSelectedBackend] = useState('wasm');
  const inputTensorBufferRef = useRef(new Float32Array(640 * 640 * 3));

  
  // shouldFlipCamera의 최신 값을 항상 참조하기 위한 ref
  const shouldFlipCameraRef = useRef(false);
  
  // isSwitchingCamera의 최신 값을 항상 참조하기 위한 ref
  const isSwitchingCameraRef = useRef(false);
  
  // customThreshold 변경 시 ref도 업데이트
  useEffect(() => {
    thresholdRef.current = customThreshold;
  }, [customThreshold]);

  // shouldFlipCamera 변경 시 ref도 업데이트
  useEffect(() => {
    shouldFlipCameraRef.current = shouldFlipCamera;
  }, [shouldFlipCamera]);

  // isSwitchingCamera 변경 시 ref도 업데이트
  useEffect(() => {
    isSwitchingCameraRef.current = isSwitchingCamera;
  }, [isSwitchingCamera]);

  // 🎯 플랫폼 감지 및 초기 설정 (한 번만 실행)
  useEffect(() => {
    // 플랫폼 감지
    const userAgent = navigator.userAgent || navigator.vendor || window.opera;
    const isMobileDevice = /android|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent.toLowerCase());
    
    setIsMobile(isMobileDevice);
    const detectedDeviceType = isMobileDevice ? "mobile" : "desktop";
    setDeviceType(detectedDeviceType);
    
    // 모바일이면 카메라 설정 조정
    if (isMobileDevice) {
      setCameraFacing("environment"); // 모바일: 후면 카메라로 시작
      setShouldFlipCamera(false); // 모바일 후면: 정방향
    }
    // PC는 이미 초기값이 올바르게 설정됨 (user, true)
    
    const finalCamera = isMobileDevice ? "environment" : "user";
    const finalFlip = isMobileDevice ? false : true;
    console.log(`✅ 플랫폼 감지 완료: ${detectedDeviceType}, 카메라: ${finalCamera}, 미러링: ${finalFlip}`);
  }, []); // 완전히 한 번만 실행

  // 카메라 타입 변경 시에만 자동 반전 재계산 (수동 설정이 없을 때만)
  useEffect(() => {
    if (manualFlip === null && deviceType) {
      if (deviceType === "desktop") {
        setShouldFlipCamera(true); // PC: 항상 미러링
      } else if (deviceType === "mobile") {
        setShouldFlipCamera(cameraFacing === "user"); // 모바일: 전면만 미러링
      }
    }
  }, [cameraFacing, deviceType, manualFlip]);
  
  // 모델 출력 구조 정보를 저장하는 ref 추가
  const modelInfoRef = useRef({
    outputShape: null,
    detectionLength: 85, // 기본값 (YOLOv5s)
    numClasses: 80,
    isYOLOv5su: false,
    coordinateFormat: 'center', // 'center' or 'corner'
    numDetections: null,
    isTransposed: false,
    isStructureDetected: false // 모델 구조 감지 완료 여부
  });

  // 모델 로드 - deviceType이 설정된 후에 실행
  useEffect(() => {
    modelLoadingStartedRef.current = false;
  }, [selectedBackend]);

  // 모델 로드 - deviceType이 설정된 후에 실행
  useEffect(() => {
    if (!deviceType || modelLoadingStartedRef.current) {
      console.log("⏳ deviceType 설정 대기 중이거나 이미 모델 로딩 시작됨...");
      return; // deviceType이 설정될 때까지 또는 이미 로딩이 시작되었으면 대기
    }
    modelLoadingStartedRef.current = true; // 🚀 로딩 시작! 플래그 설정
    
    async function loadModel() {
      try {
        setIsModelLoading(true);
        const modelPath = '/models/model.onnx';
        
        // 1. 클래스 정보 로드 (json 파일 우선, 실패 시 모델 메타데이터에서 추출)
        let classNames = [];
        try {
          const res = await fetch('/models/classes.json');
          const contentType = res.headers.get('content-type');
          if (res.ok && contentType && contentType.includes('application/json')) {
            classNames = await res.json();
            console.log("✅ `classes.json`에서 클래스 정보 로드 성공.");
          } else {
            throw new Error('`classes.json`을 찾을 수 없어 모델 메타데이터에서 추출을 시도합니다.');
          }
        } catch (e) {
          console.log(`ℹ️ ${e.message}`);
          classNames = await extractClassesFromModel(modelPath);
          if (classNames.length === 0) {
            const errorMsg = '클래스 정보를 찾을 수 없습니다. `public/models/classes.json` 파일이 있거나 모델 메타데이터에 "names" 정보가 포함되어 있는지 확인하세요.';
            console.error(`❌ ${errorMsg}`);
            setErrorMessage(errorMsg);
            setIsModelLoading(false);
            return;
          }
        }
        setClasses(classNames);
        
        // 2. ONNX 모델 및 세션 생성
        // 📱 디바이스 종류에 따라 최적의 실행 백엔드 목록을 선택
        /*const providers = deviceType === 'desktop'
          ? ['webgpu', 'webgl', 'cpu']
          : ['webgl', 'cpu'];

        console.log(`🚀 ${deviceType} 디바이스에서 모델 로딩 시작... (Providers: ${providers.join(', ')})`);
        
        const session = await ort.InferenceSession.create(modelPath, {
          executionProviders: providers,
          graphOptimizationLevel: 'all',
          enableCpuMemArena: false,
          enableMemPattern: false,
        });
        sessionRef.current = session;
        
        // 모델의 출력 구조 정보 확인
        console.log(`✅ ${deviceType} 디바이스에서 모델 로딩 완료`);
        
        // 첫 번째 실제 추론에서 자동으로 모델 구조를 감지합니다
        modelInfoRef.current.isStructureDetected = false;  */

        // 🎯 사용자가 선택한 백엔드에 따라 실행 Provider 결정
        const providerLabels = {
          webgpu: 'WEBGPU',
          webgl: 'WEBGL',
          wasm: 'CPU (WASM)',
        };

        const selectedProvider = selectedBackend;
        let session = null;
        let currentEP = 'UNKNOWN';

        try {
          console.log(`🚀 [ONNX] ${selectedProvider} 백엔드로 세션 생성 시도 중...`);

          session = await ort.InferenceSession.create(modelPath, {
            executionProviders: [selectedProvider],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: false,
            enableMemPattern: false,
          });

          currentEP = providerLabels[selectedProvider] || selectedProvider.toUpperCase();
          console.log(`✅ [ONNX] ${currentEP} 백엔드 연결 성공!`);
        } catch (err) {
          console.error(`❌ [ONNX] ${selectedProvider} 백엔드 로드 실패:`, err);
          throw err;
        }

        sessionRef.current = session;
        setActiveBackend(currentEP);

        // 첫 번째 실제 추론에서 자동으로 모델 구조를 감지합니다
        modelInfoRef.current.isStructureDetected = false;
        
        setIsModelLoaded(true);
      } catch (e) {
        console.error('Error loading model:', e);
        setErrorMessage('모델 로딩 오류. public/models 디렉토리에 model.onnx 파일이 있는지 확인하세요.');
      } finally {
        setIsModelLoading(false);
      }
    }
    loadModel();
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  // }, [deviceType]); // deviceType이 설정되면 모델 로딩
  }, [deviceType, selectedBackend]);



  // 렌더링 루프
  const drawLoop = useCallback(() => {
    const canvas = canvasRef.current;
    const video = webcamRef.current?.video;
    if (!canvas || !video || !isDetecting) {
      // 탐지가 중지되면 애니메이션 중지
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      return;
    }
    
    // 캔버스 크기를 비디오 해상도에 맞춤
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    // 그리기 전 클리어
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawBoundingBoxes(detectionMemoryRef.current, ctx, canvas.width, canvas.height, shouldFlipCameraRef.current);
    
    // 다음 프레임 예약
    animationRef.current = requestAnimationFrame(drawLoop);
  }, [isDetecting]); // shouldFlipCamera는 ref로 참조하므로 의존성에서 제거

  // 🚀 병렬 추론 처리를 위한 설정
  const activeInferencesRef = useRef(new Set());
  const lastSuccessfulResultTimeRef = useRef(0);
  // 📱 모바일 추론 빈도 제한을 위한 카운터
  const frameCounterRef = useRef(0);
  
  // 📊 성능 모니터링
  const performanceRef = useRef({
    lastFrameTime: Date.now(),
    frameCount: 0,
    currentFPS: 0
  });
  
  // 🚀 고성능 추론 루프 - 스마트 Session 관리
  const inferenceLoop = useCallback(async () => {
    if (!isDetecting || !webcamRef.current || !sessionRef.current || isSwitchingCameraRef.current) return;
    
    // 🛡️ 초고속 Session 관리 - 1개만 허용하되 최소 대기
    if (activeInferencesRef.current.size > 0) {
      // 최소 대기 - Session 완료를 기다림
      setTimeout(inferenceLoop, 1);
      return;
    }
    
    const video = webcamRef.current.video;
    if (video.readyState === 4) {
      // 📱 모든 디바이스에서 동일한 추론 빈도 (안정성 우선)
      // 추론 빈도 제한 제거
      
      const scaleCanvas = scaleCanvasRef.current;
      
      // 📱 모든 디바이스에서 동일한 스케일 팩터 (안정성 우선)
      const factor = 0.4; // 모든 디바이스: 40%
      
      scaleCanvas.width = Math.floor(video.videoWidth * factor);
      scaleCanvas.height = Math.floor(video.videoHeight * factor);
      const sctx = scaleCanvas.getContext('2d', { 
        willReadFrequently: true,
        alpha: false, // 투명도 처리 비활성화로 성능 향상
        desynchronized: true // 비동기 렌더링으로 성능 향상
      });
      sctx.drawImage(video, 0, 0, scaleCanvas.width, scaleCanvas.height);
      const imageData = sctx.getImageData(0, 0, scaleCanvas.width, scaleCanvas.height);
      const [input, iw, ih] = await prepareInput(imageData);
      
      // 🚀 안전한 추론 시작 - 고유 ID로 관리
      const inferenceId = Date.now() + Math.random();
      activeInferencesRef.current.add(inferenceId);
      
      // ⚡ 추론 실행 
      sessionRef.current.run({ images: input }).then(results => {
        // 🎯 추론 완료 후 관리
        activeInferencesRef.current.delete(inferenceId);
        
        // 📊 고성능 결과 반영 - 통일된 FPS 제한 
        const currentTime = Date.now();
        if (currentTime - lastSuccessfulResultTimeRef.current > 16) { // 60 FPS 제한
          lastSuccessfulResultTimeRef.current = currentTime;
          
          const out = results[Object.keys(results)[0]];
          
          // 🎯 실시간 모델 구조 자동 감지 (첫 번째 추론에서만 실행)
          if (!modelInfoRef.current.isStructureDetected && out.dims.length === 3) {
            const [batch, dim1, dim2] = out.dims;
            let numDetections, detectionLength, isTransposed;
            
            // 💡 [개선된 방식] `classes.length`를 사용하여 모델 구조를 더 명확하게 판단합니다.
            const numLoadedClasses = classes.length > 0 ? classes.length : 80; // 로드된 클래스 개수 (없으면 80으로 가정)
            const expectedLengths = [
              5 + numLoadedClasses, // 표준 YOLOv5: 4(좌표) + 1(객체 점수) + 클래스 개수
              4 + numLoadedClasses, // YOLOv5 변형 (객체 점수 없음): 4(좌표) + 클래스 개수
              6                     // 후처리 포함 모델: 4(좌표) + 1(신뢰도) + 1(클래스 ID)
            ];

            const dim1IsExpected = expectedLengths.includes(dim1);
            const dim2IsExpected = expectedLengths.includes(dim2);

            if (dim2IsExpected && !dim1IsExpected) {
              // dim2가 예상 길이에 해당하므로, Non-Transposed 형태: [탐지 수, 정보 길이]
              isTransposed = false;
              detectionLength = dim2;
              numDetections = dim1;
            } else if (dim1IsExpected && !dim2IsExpected) {
              // dim1이 예상 길이에 해당하므로, Transposed 형태: [정보 길이, 탐지 수]
              isTransposed = true;
              detectionLength = dim1;
              numDetections = dim2;
            } else {
              // 모호한 경우, 기존의 크기 비교 방식으로 안전하게 처리 (Fallback)
              const reason = dim1IsExpected && dim2IsExpected ? "두 차원 모두 예상 길이에 해당" : "두 차원 모두 예상 길이와 불일치";
              console.warn(`모델 구조 감지 모호함 (${reason}). 크기 비교로 대체합니다. dims:[${dim1}, ${dim2}], expected:[${expectedLengths.join(',')}]`);
              
              if (dim2 > dim1) {
                isTransposed = true;
                numDetections = dim2;
                detectionLength = dim1;
              } else {
                isTransposed = false;
                numDetections = dim1;
                detectionLength = dim2;
              }
            }
            
            // 모델 정보 업데이트
            modelInfoRef.current.isTransposed = isTransposed;
            modelInfoRef.current.numDetections = numDetections;
            modelInfoRef.current.detectionLength = detectionLength;
            
            // 🔍 [개선된 방식] `detectionLength`와 `numLoadedClasses`를 기반으로 모델 타입 명확화
            let modelType = "Unknown";
            let coordinateFormat = "center";
            let isYOLOv5su = false; // 'su'는 객체 점수가 없는 모델을 지칭하는 내부 용어
            let numClasses = numLoadedClasses;

            if (detectionLength === 6) {
              modelType = "Post-processed (Corner)";
              coordinateFormat = "corner";
              isYOLOv5su = true; // 이 형식은 객체 점수가 없음
            } else if (detectionLength === 4 + numLoadedClasses) {
              modelType = `No-Objectness (Center, ${numLoadedClasses} classes)`;
              coordinateFormat = "center";
              isYOLOv5su = true; // 객체 점수가 없음
            } else if (detectionLength === 5 + numLoadedClasses) {
              modelType = `Standard (Center, ${numLoadedClasses} classes)`;
              coordinateFormat = "center";
              isYOLOv5su = false; // 객체 점수가 있음
            } else {
              // 예상과 다른 detectionLength를 가진 커스텀 모델 처리
              const estimatedClasses = Math.max(1, detectionLength - 4);
              modelType = `Custom Model (${detectionLength} features)`;
              coordinateFormat = "center"; // 가장 흔한 형식으로 추정
              isYOLOv5su = true; // 객체 점수가 없는 형식으로 안전하게 추정
              numClasses = estimatedClasses;
              console.warn(`알 수 없는 모델 구조입니다. (length: ${detectionLength}). ${estimatedClasses}개 클래스로 추정하여 처리합니다.`);
            }
            
            // 설정 적용
            modelInfoRef.current.isYOLOv5su = isYOLOv5su;
            modelInfoRef.current.coordinateFormat = coordinateFormat;
            modelInfoRef.current.numClasses = numClasses;
            modelInfoRef.current.isStructureDetected = true;
            
            console.log(`✅ 모델 자동 감지 완료: ${modelType}, Transposed: ${isTransposed}, Detections: ${numDetections}, Length: ${detectionLength}`);
          }
          
          const newDetections = processDetections(
              out.data,
              out.dims,
              iw,
              ih,
              video.videoWidth,
              video.videoHeight,
              thresholdRef.current
          );
          
          if (newDetections.length > 0) {
            detectionMemoryRef.current = newDetections.map(d => ({ ...d, missed: 0 }));
          } else {
            detectionMemoryRef.current = detectionMemoryRef.current
                .map(d => ({ ...d, missed: d.missed + 1 }))
                .filter(d => d.missed <= MAX_MISSED_FRAMES);
          }
          setDetections(detectionMemoryRef.current);
          
          // 📊 고성능 FPS 계산 및 업데이트
          performanceRef.current.frameCount++;
          if (currentTime - performanceRef.current.lastFrameTime >= 1000) {
            const fps = performanceRef.current.frameCount;
            performanceRef.current.currentFPS = fps;
            performanceRef.current.frameCount = 0;
            performanceRef.current.lastFrameTime = currentTime;
            
            setCurrentFPS(fps);
            
            if (deviceType === 'mobile' && debugMode) {
              console.log(`📱 모바일 FPS: ${fps}`);
            }
          }
        }
        
        // ⚡ 추론 성공 - 즉시 다음 추론 시작
        setTimeout(inferenceLoop, 1);
        
      }).catch(error => {
        // 🚫 에러 처리 - 추론 ID 정리
        activeInferencesRef.current.delete(inferenceId);
        
        if (debugMode && error.message.includes('Session')) {
          console.log('⚡ Session 충돌 감지 - 빠른 재시도');
        } else if (debugMode) {
          console.warn('추론 에러 (무시):', error.message);
        }
        
        // ⚡ 에러 발생 - 최소 지연 후 재시도
        setTimeout(inferenceLoop, 5);
      });
          } else {
        // 비디오가 준비되지 않은 경우 최소 간격으로 대기
        const interval = deviceType === 'mobile' ? 20 : 10;
        setTimeout(inferenceLoop, interval);
      }
  }, [isDetecting]);

  // isDetecting 및 isSwitchingCamera 변경시 루프 시작/중지
  useEffect(() => {
    if (isDetecting && !isSwitchingCameraRef.current) {
      // 기존 애니메이션 프레임 취소
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      // 새로운 drawLoop 시작
      requestAnimationFrame(drawLoop);
      inferenceLoop();
      console.log('🔄 추론 루프 시작 - isDetecting:', isDetecting, 'isSwitchingCamera:', isSwitchingCameraRef.current);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (isDetecting && isSwitchingCameraRef.current) {
        console.log('⏸️ 카메라 전환 중 - 추론 루프 일시 중지');
      }
    }
  }, [isDetecting, isSwitchingCamera, drawLoop, inferenceLoop]); // isSwitchingCamera는 ref 동기화를 위해 유지

  // 🎯 카메라 시작 - `react-webcam`에 스트림 관리를 위임하도록 수정
  const startCamera = useCallback(() => {
    // Webcam 컴포넌트가 렌더링되어 스트림을 자체적으로 관리하도록 `isDetecting` 상태만 변경합니다.
    // 이전에 여기서 수동으로 `getUserMedia`를 호출하던 로직은 리소스 누수를 유발할 수 있어 제거했습니다.
    setIsDetecting(true);
  }, []);

  // 🎯 카메라 전환 (모바일 전용) - Android 호환성 및 안정성 강화
  const switchCamera = async () => {
    // 🔒 이미 전환 중이거나 카메라가 꺼져있으면 무시
    if (isSwitchingCamera || !isDetecting) {
      console.log('⚠️ 카메라 전환 요청 무시 - 전환 중이거나 카메라 꺼짐');
      return;
    }

    const newFacing = cameraFacing === "user" ? "environment" : "user";
    console.log(`📷 카메라 전환: ${cameraFacing} → ${newFacing}`);

    // 카메라 전환 시작 - 로딩 상태 설정
    setIsSwitchingCamera(true);

    // 🎯 이전 카메라의 모든 탐지 결과 클리어
    setDetections([]);
    detectionMemoryRef.current = [];
    console.log('🧹 카메라 전환 시 이전 탐지 결과 클리어');

    // 수동 반전 설정 리셋
    setManualFlip(null);

    // 1. 현재 스트림을 먼저 명시적으로 중지. (Android 호환성)
    if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
    }
    
    // 2. 상태 업데이트를 setTimeout으로 감싸서 스트림 중지 후 약간의 지연을 줍니다.
    //    이는 브라우저가 카메라 리소스를 해제할 시간을 확보하여 `NotReadableError`를 방지하기 위함입니다.
    setTimeout(() => {
      setCameraFacing(newFacing);
      // key를 변경하여 Webcam 컴포넌트를 완전히 새로 마운트하도록 강제합니다.
      setWebcamKey(Date.now());
    }, 100); // 안드로이드 기기에서의 안정성을 위해 100ms 지연
  };

  // 🎯 수동 반전 토글
  const toggleManualFlip = () => {
    // 🔒 카메라 전환 중이면 무시
    if (isSwitchingCamera) {
      console.log('⚠️ 미러링 토글 요청 무시 - 카메라 전환 중');
      return;
    }
    
    const newManualFlip = manualFlip === null ? !shouldFlipCamera : (manualFlip ? false : true);
    setManualFlip(newManualFlip);
    setShouldFlipCamera(newManualFlip); // 즉시 반전 상태 업데이트
    console.log(`🔄 [${deviceType}] 수동 반전 토글:`);
    console.log(`   - manualFlip: ${manualFlip} → ${newManualFlip}`);
    console.log(`   - shouldFlipCamera: ${shouldFlipCamera} → ${newManualFlip}`);
    console.log(`   - cameraFacing: ${cameraFacing}`);
    console.log(`   - ref값: ${shouldFlipCameraRef.current}`);
  };

  // 카메라 중지
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setIsDetecting(false);
    setDetections([]);
    setCurrentFPS(0); // 📊 FPS 리셋
  };

    // Process detections from model output - 수정된 버전
  const processDetections = (data, dims, imgWidth, imgHeight, canvasWidth, canvasHeight, threshold) => {
    const detectionLength = modelInfoRef.current.detectionLength;
    const numDetections = modelInfoRef.current.numDetections || dims[1];
    const isYOLOv5su = modelInfoRef.current.isYOLOv5su;
    const coordinateFormat = modelInfoRef.current.coordinateFormat;
    const isTransposed = modelInfoRef.current.isTransposed || false;

    // 모델 입력 크기 (YOLOv5는 일반적으로 640x640 입력을 사용)
    const modelWidth = 640;
    const modelHeight = 640;

    if (debugMode) {
      console.log(`=== Detection Processing ===`);
      console.log(`Canvas: ${canvasWidth}x${canvasHeight}, Model: ${modelWidth}x${modelHeight}`);
      console.log(`Model type: ${isYOLOv5su ? 'YOLOv5su' : 'YOLOv5s'}, Coordinate format: ${coordinateFormat}`);
      console.log(`Detection length: ${detectionLength}, Num detections: ${numDetections}`);
      console.log(`Output dimensions: [${dims.join(', ')}], Transposed: ${isTransposed}`);
      console.log(`Classes count: ${classes.length}, Confidence threshold: ${threshold}`);
    }

    const rawDetections = [];
    const classDetections = {};
    
    // 🔍 Custom 모델 디버깅: confidence 통계
    let processedCount = 0;
    let aboveThresholdCount = 0;

    // transpose된 데이터에서 값을 가져오는 헬퍼 함수
    const getValue = (detectionIndex, valueIndex) => {
      if (isTransposed) {
        // [1, 84, 8400] 형태에서: data[valueIndex * numDetections + detectionIndex]
        return data[valueIndex * numDetections + detectionIndex];
      } else {
        // [1, 8400, 84] 형태에서: data[detectionIndex * detectionLength + valueIndex]
        return data[detectionIndex * detectionLength + valueIndex];
      }
    };

    for (let i = 0; i < numDetections; i++) {
      let bbox, confidence, classIndex, className;

      if (isYOLOv5su) {
        if (coordinateFormat === 'corner' && detectionLength === 6) {
          // YOLOv5su corner format: [x1, y1, x2, y2, confidence, class_id]
          let x1 = getValue(i, 0);
          let y1 = getValue(i, 1);
          let x2 = getValue(i, 2);
          let y2 = getValue(i, 3);
          confidence = getValue(i, 4);
          classIndex = Math.round(getValue(i, 5));
          
          // confidence 값이 sigmoid를 거쳐야 하는지 확인
          if (confidence > 1) {
            confidence = 1 / (1 + Math.exp(-confidence));
          }
          
          if (debugMode && i < 5) {
            console.log(`Raw detection ${i}: [${x1.toFixed(4)}, ${y1.toFixed(4)}, ${x2.toFixed(4)}, ${y2.toFixed(4)}, ${confidence.toFixed(4)}, ${classIndex}]`);
          }
          
          processedCount++;
          if (confidence >= threshold) aboveThresholdCount++;
          
          // 좌표가 0-1 범위인지 확인 (정규화된 좌표)
          if (x1 <= 1 && y1 <= 1 && x2 <= 1 && y2 <= 1 && x1 >= 0 && y1 >= 0 && x2 >= 0 && y2 >= 0) {
            // 정규화된 좌표를 캔버스 크기로 변환
            x1 *= canvasWidth;
            y1 *= canvasHeight;
            x2 *= canvasWidth;
            y2 *= canvasHeight;
          } else {
            // 이미 픽셀 좌표인 경우 모델 크기에서 캔버스 크기로 스케일링
            const scaleX = canvasWidth / modelWidth;
            const scaleY = canvasHeight / modelHeight;
            x1 *= scaleX;
            y1 *= scaleY;
            x2 *= scaleX;
            y2 *= scaleY;
          }
          
          // x1,y1,x2,y2를 x,y,width,height 형태로 변환
          const x = Math.min(x1, x2);
          const y = Math.min(y1, y2);
          const width = Math.abs(x2 - x1);
          const height = Math.abs(y2 - y1);
          
          // 유효성 검사 - 너무 작거나 화면 밖의 박스 제거
          if (width < 5 || height < 5 || x < -canvasWidth || y < -canvasHeight || 
              x > canvasWidth * 2 || y > canvasHeight * 2) {
            if (debugMode && i < 3) {
              console.log(`Invalid bbox rejected: [${x.toFixed(2)}, ${y.toFixed(2)}, ${width.toFixed(2)}, ${height.toFixed(2)}]`);
            }
            continue;
          }
          
          bbox = [x, y, width, height];
          className = classes[classIndex] || `클래스 ${classIndex}`;
          
        } else if (coordinateFormat === 'center') {
          // YOLOv5su center format: [cx, cy, w, h, class1, ..., classN] (84개)
          let cx = getValue(i, 0);
          let cy = getValue(i, 1);
          let width = getValue(i, 2);
          let height = getValue(i, 3);
          
          // 클래스 확률 계산 (objectness 없음)
          let maxClassProb = 0;
          classIndex = 0;
          const numClasses = modelInfoRef.current.numClasses;
          
          for (let j = 4; j < Math.min(4 + numClasses, detectionLength); j++) {
            const prob = getValue(i, j);
            // 이미 sigmoid가 적용된 값일 가능성이 높음
            const finalProb = prob > 1 ? (1 / (1 + Math.exp(-prob))) : prob;
            if (finalProb > maxClassProb) {
              maxClassProb = finalProb;
              classIndex = j - 4;
            }
          }
          
          confidence = maxClassProb;
          className = classes[classIndex] || `클래스 ${classIndex}`;
          
          if (debugMode && i < 5) {
            console.log(`Raw center detection ${i}: [${cx.toFixed(4)}, ${cy.toFixed(4)}, ${width.toFixed(4)}, ${height.toFixed(4)}], conf=${confidence.toFixed(4)}, class=${classIndex}`);
          }
          
          processedCount++;
          if (confidence >= threshold) aboveThresholdCount++;
          
          // 좌표 변환 (정규화 여부 확인)
          if (cx <= 1 && cy <= 1 && width <= 1 && height <= 1 && cx >= 0 && cy >= 0) {
            // 정규화된 좌표
            cx *= canvasWidth;
            cy *= canvasHeight;
            width *= canvasWidth;
            height *= canvasHeight;
          } else {
            // 픽셀 좌표
            const scaleX = canvasWidth / modelWidth;
            const scaleY = canvasHeight / modelHeight;
            cx *= scaleX;
            cy *= scaleY;
            width *= scaleX;
            height *= scaleY;
          }
          
          const x = cx - width/2;
          const y = cy - height/2;
          
          // 유효성 검사
          if (width < 5 || height < 5 || x < -canvasWidth || y < -canvasHeight || 
              x > canvasWidth * 2 || y > canvasHeight * 2) {
            if (debugMode && i < 3) {
              console.log(`Invalid center bbox rejected: [${x.toFixed(2)}, ${y.toFixed(2)}, ${width.toFixed(2)}, ${height.toFixed(2)}]`);
            }
            continue;
          }
          
          bbox = [x, y, width, height];
          
          if (debugMode && i < 3) {
            console.log(`Processed center detection ${i}: bbox=[${x.toFixed(2)}, ${y.toFixed(2)}, ${width.toFixed(2)}, ${height.toFixed(2)}], conf=${confidence.toFixed(3)}, class=${classIndex}(${className})`);
          }
        }
        
      } else {
        // YOLOv5s 형태: [cx, cy, w, h, objectness, class1, ..., class80]
        const rawObjectness = getValue(i, 4);
        const objectness = 1 / (1 + Math.exp(-rawObjectness));
        let maxClassProb = 0;
        classIndex = 0;
        
        const numClasses = modelInfoRef.current.numClasses;
        for (let j = 5; j < Math.min(5 + numClasses, detectionLength); j++) {
          const rawProb = getValue(i, j);
          const prob = 1 / (1 + Math.exp(-rawProb));
        if (prob > maxClassProb) {
          maxClassProb = prob;
          classIndex = j - 5;
        }
      }
        
        confidence = objectness * maxClassProb;

        // 🔧 클래스 인덱스 범위 체크 및 fallback
        if (classIndex >= classes.length) {
          className = `클래스_${classIndex}`;
        } else {
          className = classes[classIndex] || `클래스_${classIndex}`;
        }
        
        if (debugMode && i < 5) {
          console.log(`Raw YOLOv5s detection ${i}: rawObj=${rawObjectness.toFixed(4)}, obj=${objectness.toFixed(4)}, maxClassProb=${maxClassProb.toFixed(4)}, conf=${confidence.toFixed(4)}, class=${classIndex}(${className})`);
        }
        
        processedCount++;
        if (confidence >= threshold) aboveThresholdCount++;

        // Extract bounding box coordinates (center format)
        let cx = getValue(i, 0);
        let cy = getValue(i, 1);
        let width = getValue(i, 2);
        let height = getValue(i, 3);
        
        // 🔧 좌표가 정규화되어 있는지 확인 (Custom 모델 대응)
        if (cx <= 1 && cy <= 1 && width <= 1 && height <= 1 && cx >= 0 && cy >= 0) {
          // 정규화된 좌표 (0~1 범위)
          cx *= canvasWidth;
          cy *= canvasHeight;
          width *= canvasWidth;
          height *= canvasHeight;
        } else {
          // 이미 픽셀 좌표인 경우 모델 크기에서 캔버스 크기로 스케일링
          const scaleX = canvasWidth / modelWidth;
          const scaleY = canvasHeight / modelHeight;
          cx *= scaleX;
          cy *= scaleY;
          width *= scaleX;
          height *= scaleY;
        }
        
        // center format을 corner format으로 변환
        const x = cx - width/2;
        const y = cy - height/2;
        
        // 유효성 검사
        if (width < 5 || height < 5 || x < -canvasWidth || y < -canvasHeight || 
            x > canvasWidth * 2 || y > canvasHeight * 2) {
          if (debugMode && i < 3) {
            console.log(`Invalid YOLOv5s bbox rejected: [${x.toFixed(2)}, ${y.toFixed(2)}, ${width.toFixed(2)}, ${height.toFixed(2)}]`);
          }
          continue;
        }
        
        bbox = [x, y, width, height];
        
        if (debugMode && i < 3) {
          console.log(`Processed YOLOv5s detection ${i}: bbox=[${x.toFixed(2)}, ${y.toFixed(2)}, ${width.toFixed(2)}, ${height.toFixed(2)}], conf=${confidence.toFixed(3)}, class=${classIndex}(${className})`);
        }
      }

      // Filter by confidence threshold
      if (confidence > threshold) {
        // Track detections by class
        if (!classDetections[className]) {
          classDetections[className] = {
            count: 0,
            confidenceSum: 0,
            highestConfidence: 0
          };
        }
        classDetections[className].count++;
        classDetections[className].confidenceSum += confidence;
        classDetections[className].highestConfidence = Math.max(classDetections[className].highestConfidence, confidence);

        // Log individual detection with high confidence (only in debug mode)
        if (debugMode && rawDetections.length < 5) {
          console.log(`탐지된 객체: ${className}, 인식률: ${(confidence * 100).toFixed(2)}%, 위치: [x=${Math.round(bbox[0])}, y=${Math.round(bbox[1])}, w=${Math.round(bbox[2])}, h=${Math.round(bbox[3])}]`);
        }

        rawDetections.push({
          bbox: bbox,
          class: className,
          confidence: confidence,
          classIndex: classIndex
        });
      }
    }

    // Apply Non-Maximum Suppression to filter out overlapping detections
    const detections = applyNonMaxSuppression(rawDetections, 0.3); // 0.3 is the IoU threshold (lower value = more aggressive filtering)

    // 🔍 Custom 모델 디버깅: 처리 통계 출력
    if (debugMode) {
      if (Object.keys(classDetections).length > 0) {
        console.log('======== 추론 결과 요약 ========');
        console.log(`총 탐지된 객체 수: ${rawDetections.length}개 (NMS 적용 전), ${detections.length}개 (NMS 적용 후)`);

        // 클래스별 탐지 결과 및 인식률 출력
        Object.keys(classDetections).forEach(className => {
          const stats = classDetections[className];
          const avgConfidence = stats.confidenceSum / stats.count;
          console.log(`- ${className}: ${stats.count}개, 평균 인식률: ${(avgConfidence * 100).toFixed(2)}%, 최고 인식률: ${(stats.highestConfidence * 100).toFixed(2)}%`);
        });
        console.log('===============================');
      } else {
        console.log('⚠️ 탐지된 객체가 없습니다.');
        
        // 가능한 원인 제시
        if (rawDetections.length === 0) {
          console.log(`   (i) 현재 신뢰도(${Math.round(threshold*100)}%)에서 탐지된 객체가 없습니다. 신뢰도를 조정해보세요.`);
        }
      }
    }

    return detections;
  };

  // Calculate Intersection over Union (IoU) between two bounding boxes
  const calculateIoU = (box1, box2) => {
    // box format: [x, y, width, height] where x,y is the top-left corner
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;

    // Calculate coordinates of the intersection rectangle
    const xLeft = Math.max(x1, x2);
    const yTop = Math.max(y1, y2);
    const xRight = Math.min(x1 + w1, x2 + w2);
    const yBottom = Math.min(y1 + h1, y2 + h2);

    // Check if there is an intersection
    if (xRight < xLeft || yBottom < yTop) {
      return 0;
    }

    // Calculate intersection area
    const intersectionArea = (xRight - xLeft) * (yBottom - yTop);

    // Calculate union area
    const box1Area = w1 * h1;
    const box2Area = w2 * h2;
    const unionArea = box1Area + box2Area - intersectionArea;

    // Calculate IoU
    return intersectionArea / unionArea;
  };

  // Apply Non-Maximum Suppression to filter out overlapping detections
  const applyNonMaxSuppression = (detections, iouThreshold) => {
    if (detections.length === 0) {
      return [];
    }

    // Sort detections by confidence score (highest first)
    const sortedDetections = [...detections].sort((a, b) => b.confidence - a.confidence);

    const selectedDetections = [];
    const remainingDetections = [...sortedDetections];

    // Process all detections
    while (remainingDetections.length > 0) {
      // Select the detection with highest confidence
      const currentDetection = remainingDetections.shift();
      selectedDetections.push(currentDetection);

      // Filter out detections that have high overlap with the current detection
      // and are of the same class
      let i = 0;
      while (i < remainingDetections.length) {
        // Only apply NMS to detections of the same class
        if (remainingDetections[i].classIndex === currentDetection.classIndex) {
          const iou = calculateIoU(currentDetection.bbox, remainingDetections[i].bbox);

          // If IoU is above threshold, remove this detection
          if (iou > iouThreshold) {
            remainingDetections.splice(i, 1);
          } else {
            i++;
          }
        } else {
          i++;
        }
      }
    }

    return selectedDetections;
  };

  // 📱 고성능 모바일 최적화 input 준비
  const prepareInput = async (imageData) => {
    const imgWidth = imageData.width;
    const imgHeight = imageData.height;

    // 📱 모든 디바이스에서 동일한 모델 입력 크기 사용 (안정성 우선)
    const modelWidth = 640;  // 모든 디바이스: 640x640
    const modelHeight = 640; // 모든 디바이스: 640x640

    // 재사용 가능한 canvas들 (성능 최적화)
    const sourceCanvas = sourceCanvasRef.current;
    sourceCanvas.width = imgWidth;
    sourceCanvas.height = imgHeight;
    const sourceCtx = sourceCanvas.getContext('2d', { 
      willReadFrequently: true,
      alpha: false 
    });
    sourceCtx.putImageData(imageData, 0, 0);

    const tempCanvas = tempCanvasRef.current;
    tempCanvas.width = modelWidth;
    tempCanvas.height = modelHeight;
    const tempCtx = tempCanvas.getContext('2d', { 
      willReadFrequently: true,
      alpha: false,
      desynchronized: true 
    });

    // 이미지 리사이징
    tempCtx.drawImage(
        sourceCanvas,
        0, 0, imgWidth, imgHeight,
        0, 0, modelWidth, modelHeight
    );

    const resizedImageData = tempCtx.getImageData(0, 0, modelWidth, modelHeight);
    const pixels = resizedImageData.data;

    // 📱 모바일 최적화된 픽셀 데이터 변환 (더 효율적인 방법)
    const totalPixels = modelWidth * modelHeight;
    //const inputTensor = new Float32Array(totalPixels * 3);
    // 🚀 [수정] 매 프레임 새 배열 생성(GC 유발) 대신 미리 만든 버퍼 재사용
    const inputTensor = inputTensorBufferRef.current;
    
    // 한번에 RGB 채널별로 처리 (캐시 친화적)
    for (let i = 0; i < totalPixels; i++) {
      const pixelIndex = i * 4; // RGBA
      inputTensor[i] = pixels[pixelIndex] / 255.0;           // R
      inputTensor[i + totalPixels] = pixels[pixelIndex + 1] / 255.0;     // G  
      inputTensor[i + totalPixels * 2] = pixels[pixelIndex + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor('float32', inputTensor, [1, 3, modelHeight, modelWidth]);

    return [tensor, imgWidth, imgHeight];
  };

  // Draw bounding boxes on the canvas
  const drawBoundingBoxes = (detections, ctx, canvasWidth, canvasHeight, isFlipped = false) => {
    try {
      // 캔버스 상태 저장
      ctx.save();

      // 🎯 반전 처리를 여기서 하지 않고, 개별 요소별로 처리

      // 기본 캔버스 설정 (모바일 호환성 위해 간소화)
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'medium'; // high에서 medium으로 조정

      if (detections.length > 0) {
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      }

      // 모든 탐지된 객체 사용 (필터링 없음)
      const recentDetections = detections;

      // 비디오 프레임은 더 이상 캔버스에 그리지 않음 (Webcam 컴포넌트가 배경으로 표시됨)
      if (debugMode && webcamRef.current && webcamRef.current.video) {
        const video = webcamRef.current.video;
        if (video.readyState === 4) {
          // Log video dimensions only in debug mode
          console.log(`Video dimensions: ${video.videoWidth}x${video.videoHeight}`);
        }
      }

      if (debugMode) {
        console.log(`Drawing ${recentDetections.length} bounding boxes - Canvas dimensions: ${canvasWidth}x${canvasHeight}`);
      }

      // 최종 화면에 표시되는 탐지 결과 요약 (디버그 모드에서만)
      if (debugMode && recentDetections.length > 0) {
        console.log('======== 화면 표시 탐지 결과 ========');
        // 클래스별 통계 계산을 위한 객체
        const finalClassStats = {};

        recentDetections.forEach(detection => {
          const className = detection.class;
          if (!finalClassStats[className]) {
            finalClassStats[className] = {
              count: 0,
              confidenceSum: 0
            };
          }
          finalClassStats[className].count++;
          finalClassStats[className].confidenceSum += detection.confidence;
        });

        // 클래스별 통계 출력
        Object.keys(finalClassStats).forEach(className => {
          const stats = finalClassStats[className];
          const avgConfidence = stats.confidenceSum / stats.count;
          console.log(`클래스: ${className}, 개수: ${stats.count}개, 평균 인식률: ${(avgConfidence * 100).toFixed(2)}%`);
        });
        console.log('==================================');
      }

      // Set drawing styles
      ctx.lineWidth = 1; // Reduced line width as requested
      ctx.font = 'bold 18px Arial'; // Made font bold for better visibility
      ctx.textBaseline = 'top';

      // Draw each filtered detection
      recentDetections.forEach((detection, i) => {
        let [drawX, drawY, width, height] = detection.bbox;
        const className = detection.class;
        const confidence = detection.confidence;
        const classIndex = detection.classIndex;

        // 🎯 반전 상태에 따른 좌표 변환
        if (isFlipped) {
          drawX = canvasWidth - drawX - width; // X 좌표 반전
          if (debugMode) {
            console.log(`미러링 적용: 원본 X=${detection.bbox[0]}, 변환된 X=${drawX}`);
          }
        }

        // Log the detection information with more details in Korean (only in debug mode)
        if (debugMode) {
          console.log(`최종 탐지 #${i+1}: 클래스=${className}, 인식률=${(confidence * 100).toFixed(2)}%, 위치=[x=${Math.round(drawX)}, y=${Math.round(drawY)}, 너비=${Math.round(width)}, 높이=${Math.round(height)}]`);
        }

        // Generate random color based on class index
        const hue = (classIndex * 137) % 360; // Use prime number for better distribution

        // 더 얇은 테두리로 바운딩 박스 그리기
        ctx.strokeStyle = `hsl(${hue}, 100%, 50%)`;
        ctx.lineWidth = 1;

        // Draw bounding box with more visible style
        ctx.beginPath();
        ctx.rect(drawX, drawY, width, height);
        ctx.stroke();

        // 라벨 배경도 약간 투명하게 조정
        const label = `${className} ${Math.round(confidence * 100)}%`;
        const textMetrics = ctx.measureText(label);
        const textHeight = 28; // 라벨 높이 유지
        ctx.fillStyle = `hsla(${hue}, 100%, 50%, 0.7)`; // 약간 투명하게 조정

        // 🎯 텍스트는 항상 바운딩박스 왼쪽 위에 표시 (좌표는 이미 변환됨)
        const labelX = drawX;
        const labelY = drawY - textHeight;

        // 배경 그리기
        ctx.fillRect(
            labelX,
            labelY,
            textMetrics.width + 10,
            textHeight
        );

        // 🎯 텍스트는 반전 없이 정상적으로 그리기 (좌표만 변환된 상태)
        ctx.shadowColor = 'black';
        ctx.shadowBlur = 3;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;

        // 흰색 굵은 텍스트
        ctx.fillStyle = 'white';
        ctx.font = 'bold 18px Arial';
        ctx.fillText(label, labelX + 5, labelY + 5);

        // 그림자 효과 제거
        ctx.shadowColor = 'transparent';
      });

      // Restore the canvas state
      ctx.restore();

    } catch (error) {
      console.error("Error in drawBoundingBoxes:", error);
    }
  };

  return (
      <div className="app-container">
        {/* Header */}
        <div className="header">
          <div className="header-content">
            <div className="logo-container">
              <div>
                <Icon name="logo" />
              </div>
              <div>
                <h1>D-Lab Flow</h1>
                <p className="subtitle">실시간 객체 탐지 데모</p>
              </div>
            </div>

            {/* Status Indicators */}
            <div className="status-indicators" style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'stretch',
              gap: '0.5rem',
              width: deviceType === 'mobile' ? '200px' : '150px', // 모바일에서 너비 축소
              flexShrink: 0
            }}>
              <ConfidenceSlider
                initialThreshold={customThreshold}
                onThresholdChange={setCustomThreshold}
                disabled={!isModelLoaded || isModelLoading}
              />
              
              {/* 🎯 카메라 상태 표시 - 항상 영역 유지 */}
              <div style={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '0.5rem',
                width: '100%',
                fontSize: '0.75rem',
                color: 'var(--text-secondary)'
              }}>
                <label
                    htmlFor="backend-select"
                    style={{
                      whiteSpace: 'nowrap',
                      flexShrink: 0
                    }}
                >
                  가속 엔진
                </label>
                <select
                    id="backend-select"
                    value={selectedBackend}
                    disabled={isModelLoading || isDetecting}
                    onChange={(e) => {
                      setIsModelLoaded(false);
                      setActiveBackend('확인 중...');
                      setSelectedBackend(e.target.value);
                    }}
                    style={{
                      flex: 1,
                      minWidth: 0,
                      padding: '0.25rem',
                      borderRadius: '0.25rem',
                      border: '1px solid rgba(148, 163, 184, 0.4)',
                      backgroundColor: 'var(--background-color)',
                      color: 'var(--text-primary)',
                      fontSize: '0.75rem'
                    }}
                >
                  <option value="wasm">CPU</option>
                  <option value="webgl">WebGL</option>
                  <option value="webgpu">WebGPU</option>
                </select>
              </div>

              <div className="camera-status-info" style={{ 
                fontSize: '0.75rem', 
                color: 'var(--text-secondary)',
                textAlign: 'right',
                lineHeight: '1.2',
                minHeight: '1.8em', // 버튼을 위해 높이 증가
                width: '100%', // 부모 너비에 맞춤
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                overflow: 'visible', // 버튼이 보이도록 변경
                whiteSpace: 'nowrap' // 텍스트 줄바꿈 방지
              }}>
                {isDetecting && (
                  <>
                    {deviceType === "mobile" ? "📱" : "🖥️"} {deviceType === "mobile" ? "모바일" : "PC"}
                    {deviceType === "mobile" && (
                      <>
                        {" "}
                        <button
                          onClick={(e) => {
                            if (!isSwitchingCamera) {
                              switchCamera();
                            }
                            // 클릭 후 즉시 focus 제거
                            e.target.blur();
                          }}
                          disabled={isSwitchingCamera}
                          onTouchEnd={(e) => {
                            // 모바일 터치 이벤트에서 focus 제거
                            setTimeout(() => {
                              e.target.blur();
                              // body에 focus를 이동하여 버튼 focus 완전 제거
                              if (document.body.focus) {
                                document.body.focus();
                              } else {
                                document.activeElement?.blur();
                              }
                            }, 10);
                          }}
                          style={{
                            all: 'unset',
                            display: 'inline-block',
                            backgroundColor: isSwitchingCamera ? 'rgba(128, 128, 128, 0.1)' : 'rgba(59, 130, 246, 0.1)',
                            border: isSwitchingCamera ? '1px solid rgba(128, 128, 128, 0.3)' : '1px solid rgba(59, 130, 246, 0.3)',
                            borderRadius: '0.25rem',
                            color: isSwitchingCamera ? 'var(--text-secondary)' : 'var(--primary-color)',
                            cursor: isSwitchingCamera ? 'not-allowed' : 'pointer',
                            fontSize: '0.7rem',
                            padding: '0.2rem 0.4rem',
                            margin: '0 0.2rem',
                            transition: 'all 0.2s ease',
                            fontWeight: '500',
                            outline: 'none',
                            boxShadow: 'none',
                            WebkitTapHighlightColor: 'transparent',
                            tabIndex: '-1',
                            opacity: isSwitchingCamera ? 0.6 : 1
                          }}
                          onMouseEnter={(e) => {
                            // PC에서만 호버 효과 (비활성화 상태가 아닐 때만)
                            if (deviceType === "desktop" && !isSwitchingCamera) {
                              e.target.style.backgroundColor = 'rgba(59, 130, 246, 0.2)';
                              e.target.style.borderColor = 'rgba(59, 130, 246, 0.5)';
                            }
                          }}
                          onMouseLeave={(e) => {
                            // PC에서만 호버 효과 해제 (비활성화 상태가 아닐 때만)
                            if (deviceType === "desktop" && !isSwitchingCamera) {
                              e.target.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
                              e.target.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                            }
                          }}
                          title={isSwitchingCamera ? "카메라 전환 중..." : "클릭하여 카메라 전환"}
                        >
                          📷 {cameraFacing === "user" ? "전면" : "후면"}
                        </button>
                      </>
                    )}
                    {" "}
                    <button 
                      onClick={(e) => {
                        if (!isSwitchingCamera) {
                          toggleManualFlip();
                        }
                        e.target.blur(); // 클릭 후 focus 제거
                      }}
                      disabled={isSwitchingCamera}
                      onTouchEnd={(e) => {
                        // 모바일 터치 이벤트에서 focus 제거 (preventDefault 제거)
                        setTimeout(() => {
                          e.target.blur();
                          // body에 focus를 이동하여 버튼 focus 완전 제거
                          if (document.body.focus) {
                            document.body.focus();
                          } else {
                            document.activeElement?.blur();
                          }
                        }, 10);
                      }}
                      style={{
                        backgroundColor: isSwitchingCamera ? 'rgba(128, 128, 128, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                        border: isSwitchingCamera ? '1px solid rgba(128, 128, 128, 0.3)' : '1px solid rgba(16, 185, 129, 0.3)',
                        borderRadius: '0.25rem',
                        color: isSwitchingCamera ? 'var(--text-secondary)' : 'var(--primary-color)',
                        cursor: isSwitchingCamera ? 'not-allowed' : 'pointer',
                        fontSize: '0.7rem',
                        padding: '0.2rem 0.4rem',
                        margin: '0 0.2rem',
                        transition: 'all 0.2s ease',
                        fontWeight: '500',
                        outline: 'none', // focus outline 제거
                        WebkitTapHighlightColor: 'transparent', // 모바일 터치 하이라이트 제거
                        tabIndex: '-1', // 탭 네비게이션에서 제외
                        opacity: isSwitchingCamera ? 0.6 : 1
                      }}
                      onMouseEnter={(e) => {
                        // PC에서만 호버 효과 적용 (비활성화 상태가 아닐 때만)
                        if (deviceType === "desktop" && !isSwitchingCamera) {
                          e.target.style.backgroundColor = 'rgba(16, 185, 129, 0.2)';
                          e.target.style.borderColor = 'rgba(16, 185, 129, 0.5)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        // PC에서만 호버 효과 해제 (비활성화 상태가 아닐 때만)
                        if (deviceType === "desktop" && !isSwitchingCamera) {
                          e.target.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
                          e.target.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                        }
                      }}
                      title={isSwitchingCamera ? "카메라 전환 중..." : "클릭하여 화면 방향 전환"}
                    >
                      🔄 {shouldFlipCamera ? "미러링" : "정방향"}
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="main-content">
          {isModelLoading ? (
              <Card className="loading-card">
                <div className="loading-spinner">
                  <Icon name="loader" />
                </div>
                <h2>AI 모델 로딩 중...</h2>
              </Card>
          ) : errorMessage ? (
              <Card className="error-card">
                <div className="error-icon">
                  <Icon name="power" />
                </div>
                <h2>오류 발생</h2>
                <p>{errorMessage}</p>
                <Button
                    onClick={() => window.location.reload()}
                    className="start-button"
                    style={{ marginTop: '1rem', background: 'var(--danger-color)' }}
                >
                  다시 시도
                </Button>
              </Card>
          ) : !isDetecting ? (
              <StartCard onStart={startCamera} isModelLoaded={isModelLoaded} />
          ) : (
              <div className="camera-view">
                {/* Camera View */}
                <div 
                  className="camera-container"
                >
                  <Webcam
                      key={webcamKey}
                      ref={webcamRef}
                      audio={false}
                      screenshotFormat="image/jpeg"
                      videoConstraints={{
                        facingMode: cameraFacing, // 🎯 동적 카메라 설정
                        // 📱 안드로이드 호환성을 위해 해상도 제약 조건 완화
                        // width: { ideal: 960, min: 640 },
                        // height: { ideal: 540, min: 480 },
                        // frameRate: { ideal: 24, max: 30 },
                        // aspectRatio: 16/9
                      }}
                      style={{
                        position: 'absolute',
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        zIndex: 1,
                        left: 0,
                        top: 0,
                        borderRadius: 'inherit',
                        transform: shouldFlipCamera ? 'scaleX(-1)' : 'none', // 🎯 비디오만 반전
                      }}
                      onUserMedia={(stream) => {
                        if (debugMode) {
                          console.log("Camera stream obtained successfully");
                          console.log("Stream settings:", stream.getVideoTracks()[0].getSettings());
                        }
                        streamRef.current = stream;

                        // 카메라 영상이 실제로 로드되었으므로 전환 로딩 상태 해제
                        setIsSwitchingCamera(false);
                        
                        // 카메라 전환 완료 후 추론 루프 명시적 재시작
                        if (isDetecting) {
                          setTimeout(() => {
                            // ref를 통해 최신 함수에 접근
                            if (typeof inferenceLoop === 'function') {
                              inferenceLoop();
                              console.log('🔄 카메라 전환 완료 - 추론 루프 재시작');
                            }
                          }, 100); // 100ms 후 재시작하여 카메라 안정화 대기
                        }

                        // 캔버스 정보 확인 (디버깅용)
                        if (debugMode) {
                          const canvas = canvasRef.current;
                          if (canvas) {
                            console.log("Canvas ready:", canvas.width, canvas.height);
                          }
                        }
                      }}
                      onUserMediaError={(error) => {
                        console.error("Camera access error:", error);
                        let errorMsg = '카메라 접근 오류';

                        if (error.name === 'NotAllowedError') {
                          errorMsg = '카메라 권한이 거부되었습니다. 브라우저 설정에서 카메라 권한을 허용해주세요.';
                        } else if (error.name === 'NotFoundError') {
                          errorMsg = '카메라를 찾을 수 없습니다.';
                        } else if (error.name === 'NotSupportedError') {
                          errorMsg = 'HTTPS 연결이 필요합니다.';
                        }

                        setErrorMessage(errorMsg);
                        setIsDetecting(false);
                      }}
                  />
                  <canvas
                      ref={canvasRef}
                      className="detection-canvas"
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        zIndex: 2,
                        objectFit: 'cover',
                        borderRadius: 'inherit',
                        touchAction: 'none',
                        pointerEvents: 'none',
                        // 🎯 캔버스는 CSS 반전 없이, 내부에서 좌표 변환으로 처리
                      }}
                  />

                  {/* 🎯 카메라 전환 로딩 오버레이 */}
                  {isSwitchingCamera && (
                    <div style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      backgroundColor: 'rgba(0, 0, 0, 0.7)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      zIndex: 3,
                      borderRadius: 'inherit'
                    }}>
                      <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        color: 'white',
                        fontSize: '1rem'
                      }}>
                        <div className="loading-spinner" style={{ marginBottom: '0.5rem' }}>
                          <Icon name="loader" />
                  </div>
                        <div>카메라 전환 중...</div>
                      </div>
                    </div>
                  )}

                  {/* 🎯 모든 컨트롤 제거 - 완전히 깔끔한 화면 */}
                </div>

                {/* Detection Info - Always show the card, even when empty */}
                <Card className="detections-card">
                  <h3 style={{ marginBottom: '0.5rem' }}>
                    {detections.length > 0
                        ? `탐지된 객체 (${detections.length})`
                      : '객체를 카메라에 비춰보세요'}
                  </h3>
                  
                                     {/* 📊 성능 정보 표시 */}
                   {/*<div style={{
                     fontSize: '0.85rem', 
                     color: 'var(--text-secondary)', 
                     marginBottom: '0.5rem'
                   }}>
                     추론 속도: <span style={{ color: 'var(--primary-color)', fontWeight: '500' }}>{currentFPS} FPS</span>
                   </div>*/}
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    fontSize: '0.85rem',
                    color: 'var(--text-secondary)',
                    marginBottom: '0.5rem'
                  }}>
                    <div>
                      가속 엔진:
                      <span style={{
                        marginLeft: '6px',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '0.75rem',
                        fontWeight: 'bold',
                        backgroundColor:
                            activeBackend === 'WEBGPU'
                                ? 'rgba(16, 185, 129, 0.2)'
                                : activeBackend === 'WEBGL'
                                    ? 'rgba(59, 130, 246, 0.2)'
                                    : 'rgba(148, 163, 184, 0.2)',
                        color:
                            activeBackend === 'WEBGPU'
                                ? '#10B981'
                                : activeBackend === 'WEBGL'
                                    ? '#3B82F6'
                                    : '#94A3B8',
                        border: `1px solid ${
                            activeBackend === 'WEBGPU'
                                ? '#10B981'
                                : activeBackend === 'WEBGL'
                                    ? '#3B82F6'
                                    : '#94A3B8'
                        }`
                      }}>
                          {activeBackend}
                        </span>
                    </div>
                    <div>
                      추론 속도: <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>{currentFPS} FPS</span>
                    </div>
                  </div>

                  {detections.length > 0 ? (
                      <div className="detection-badges">
                        {detections.map((detection, index) => (
                            <Badge
                                key={index}
                                className="detection-badge"
                                style={{
                                  // 클래스에 따라 다른 색상 적용
                                  backgroundColor: `rgba(${(index * 50) % 255}, ${(index * 120) % 255}, ${(index * 180) % 255}, 0.15)`,
                                  borderColor: `rgba(${(index * 50) % 255}, ${(index * 120) % 255}, ${(index * 180) % 255}, 0.4)`
                                }}
                            >
                              {detection.class} ({Math.round(detection.confidence * 100)}%)
                            </Badge>
                        ))}
                      </div>
                  ) : isDetecting ? (
                      <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
                        <p style={{ color: 'var(--text-secondary)' }}>
                          탐지된 객체가 없습니다
                        </p>

                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.5rem' }}>
                          카메라에 객체를 비추거나 신뢰도 설정을 조정해보세요
                        </p>
                      </div>
                  ) : (
                      <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                        카메라에 객체를 비추면 여기에 탐지 결과가 표시됩니다
                      </p>
                  )}
                </Card>
              </div>
          )}
        </div>

        {/* Footer Info */}
        <div className="footer">
          <p>💡 D-Lab Flow에서 생성한 인공지능 모델을 사용한 실시간 객체 탐지 데모</p>
        </div>
      </div>
  );
}

export default App
