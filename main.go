package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"html/template"
	"image"
	"image/color"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"gocv.io/x/gocv"
)

// アプリケーション設定を更新
const (
	port            = ":8080"
	frameDelay      = 500 * time.Millisecond // 顔検出の間隔
	displayDelay    = 33 * time.Millisecond  // 表示更新の間隔（約30FPS）
	minFaceRatio    = 0.1
	maxFaceRatio    = 0.8
	scaleFactor     = 1.1
	minNeighbors    = 2
	defaultCameraID = 0
)

// WebSocket設定
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// FaceDetector 顔検出処理を管理する構造体
type FaceDetector struct {
	classifier *gocv.CascadeClassifier
	minSize    image.Point
	maxSize    image.Point
}

// NewFaceDetector 新しいFaceDetectorインスタンスを作成
func NewFaceDetector(width, height float64) (*FaceDetector, error) {
	classifier := gocv.NewCascadeClassifier()
	if !classifier.Load("data/haarcascade_frontalface_default.xml") {
		return nil, fmt.Errorf("分類器の読み込みに失敗")
	}

	return &FaceDetector{
		classifier: &classifier,
		minSize:    image.Point{X: int(width * minFaceRatio), Y: int(height * minFaceRatio)},
		maxSize:    image.Point{X: int(width * maxFaceRatio), Y: int(height * maxFaceRatio)},
	}, nil
}

// DetectFaces 画像から顔を検出
func (fd *FaceDetector) DetectFaces(img *gocv.Mat) []image.Rectangle {
	return fd.classifier.DetectMultiScaleWithParams(
		*img,
		scaleFactor,
		minNeighbors,
		0,
		fd.minSize,
		fd.maxSize,
	)
}

// Close リソースを解放
func (fd *FaceDetector) Close() {
	fd.classifier.Close()
}

// LivenessDetector なりすまし検出器
type LivenessDetector struct {
	eyeCascade     *gocv.CascadeClassifier
	lastBlinkTime  time.Time
	blinkCount     int
	checkStartTime time.Time
	motionHistory  []image.Point
	prevFace       gocv.Mat
}

func NewLivenessDetector() (*LivenessDetector, error) {
	eyeCascade := gocv.NewCascadeClassifier()
	if !eyeCascade.Load("data/haarcascade_eye.xml") {
		return nil, fmt.Errorf("目検出用の分類器の読み込みに失敗")
	}

	return &LivenessDetector{
		eyeCascade:     &eyeCascade,
		checkStartTime: time.Now(),
		motionHistory:  make([]image.Point, 0),
		prevFace:       gocv.NewMat(),
	}, nil
}

func (ld *LivenessDetector) Close() {
	ld.eyeCascade.Close()
	ld.prevFace.Close()
}

type LivenessCheck struct {
	IsLive     bool
	Confidence float64
	Reason     string
}

func (ld *LivenessDetector) DetectLiveness(face gocv.Mat) LivenessCheck {
	// 1. テクスチャ分析
	textureScore := ld.analyzeTexture(face)

	// 2. 瞬き検出
	blinkDetected := ld.detectBlink(face)

	// 3. 顔の動き検出
	motionDetected := ld.detectMotion(face)

	// 4. 深度分析
	depthScore := ld.analyzeDepth(face)

	// 総合評価
	confidence := ld.calculateConfidence(textureScore, blinkDetected, motionDetected, depthScore)

	return LivenessCheck{
		IsLive:     confidence > 0.8,
		Confidence: confidence,
		Reason:     ld.getEvaluationReason(textureScore, blinkDetected, motionDetected, depthScore),
	}
}

func (ld *LivenessDetector) analyzeTexture(face gocv.Mat) float64 {
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(face, &gray, gocv.ColorBGRToGray)

	// より詳細なテクスチャ分析のためにLBPに似た手法を使用
	blurred := gocv.NewMat()
	defer blurred.Close()
	gocv.GaussianBlur(gray, &blurred, image.Point{X: 5, Y: 5}, 0, 0, gocv.BorderDefault)

	// エッジ検出の感度を上げる
	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(gray, &edges, 50, 150) // しきい値を調整

	// テクスチャの詳細度を計算
	diff := gocv.NewMat()
	defer diff.Close()
	gocv.AbsDiff(gray, blurred, &diff)

	mean := gocv.NewMat()
	stddev := gocv.NewMat()
	defer mean.Close()
	defer stddev.Close()

	gocv.MeanStdDev(diff, &mean, &stddev)
	textureScore := normalizeScore(stddev.GetDoubleAt(0, 0), 0, 30) // しきい値を調整

	return textureScore
}

func (ld *LivenessDetector) detectBlink(face gocv.Mat) bool {
	eyes := ld.eyeCascade.DetectMultiScale(face)

	if len(eyes) == 2 {
		// 目の状態を分析
		for _, eye := range eyes {
			eyeRegion := face.Region(eye)
			defer eyeRegion.Close()

			// 目の開閉状態を判定
			if ld.isEyeClosed(eyeRegion) {
				if time.Since(ld.lastBlinkTime) > time.Second {
					ld.blinkCount++
					ld.lastBlinkTime = time.Now()
					return true
				}
			}
		}
	}
	return false
}

func (ld *LivenessDetector) detectMotion(face gocv.Mat) bool {
	if ld.prevFace.Empty() {
		face.CopyTo(&ld.prevFace)
		return false
	}

	faceSize := face.Size()
	size := image.Point{X: faceSize[1], Y: faceSize[0]}

	resizedPrev := gocv.NewMat()
	defer resizedPrev.Close()
	gocv.Resize(ld.prevFace, &resizedPrev, size, 0, 0, gocv.InterpolationLinear)

	prevGray := gocv.NewMat()
	currGray := gocv.NewMat()
	defer prevGray.Close()
	defer currGray.Close()

	gocv.CvtColor(resizedPrev, &prevGray, gocv.ColorBGRToGray)
	gocv.CvtColor(face, &currGray, gocv.ColorBGRToGray)

	diff := gocv.NewMat()
	defer diff.Close()
	gocv.AbsDiff(prevGray, currGray, &diff)

	mean := gocv.NewMat()
	stddev := gocv.NewMat()
	defer mean.Close()
	defer stddev.Close()

	gocv.MeanStdDev(diff, &mean, &stddev)
	face.CopyTo(&ld.prevFace)

	return mean.GetDoubleAt(0, 0) > 5.0 // しきい値を下げて感度を上げる
}

func (ld *LivenessDetector) analyzeDepth(face gocv.Mat) float64 {
	gradX := gocv.NewMat()
	gradY := gocv.NewMat()
	defer gradX.Close()
	defer gradY.Close()

	gocv.Sobel(face, &gradX, gocv.MatTypeCV64F, 1, 0, 3, 1, 0, gocv.BorderDefault)
	gocv.Sobel(face, &gradY, gocv.MatTypeCV64F, 0, 1, 3, 1, 0, gocv.BorderDefault)

	magnitude := gocv.NewMat()
	defer magnitude.Close()
	gocv.Magnitude(gradX, gradY, &magnitude)

	mean := gocv.NewMat()
	stddev := gocv.NewMat()
	defer mean.Close()
	defer stddev.Close()

	gocv.MeanStdDev(magnitude, &mean, &stddev)
	return normalizeScore(mean.GetDoubleAt(0, 0), 0, 100)
}

func (ld *LivenessDetector) calculateConfidence(textureScore float64, blinkDetected, motionDetected bool, depthScore float64) float64 {
	// 重みを調整
	weights := map[string]float64{
		"texture": 0.4, // テクスチャの重要度を上げる
		"blink":   0.2,
		"motion":  0.3, // モーション検出の重要度を上げる
		"depth":   0.1,
	}

	confidence := textureScore * weights["texture"]
	if blinkDetected {
		confidence += weights["blink"]
	}
	if motionDetected {
		confidence += weights["motion"]
	}
	confidence += depthScore * weights["depth"]

	return confidence
}

func (ld *LivenessDetector) getEvaluationReason(textureScore float64, blinkDetected, motionDetected bool, depthScore float64) string {
	var reasons []string

	if textureScore < 0.5 {
		reasons = append(reasons, "Texture")
	}
	if !blinkDetected {
		reasons = append(reasons, "No Blink")
	}
	if !motionDetected {
		reasons = append(reasons, "No Motion")
	}
	if depthScore < 0.5 {
		reasons = append(reasons, "Depth")
	}

	if len(reasons) == 0 {
		return "OK"
	}
	return strings.Join(reasons, ", ")
}

// VideoHandlerを修正
type VideoHandler struct {
	webcam           *gocv.VideoCapture
	detector         *FaceDetector
	template         *template.Template
	frameSize        image.Point
	lastFaces        []image.Rectangle
	faceMutex        sync.RWMutex
	clients          map[*websocket.Conn]bool
	clientsMux       sync.RWMutex
	livenessDetector *LivenessDetector
	done             chan struct{} // 終了通知用
}

// NewVideoHandler 新しいVideoHandlerインスタンスを作成
func NewVideoHandler() (*VideoHandler, error) {
	webcam, err := gocv.VideoCaptureDevice(defaultCameraID)
	if err != nil {
		return nil, fmt.Errorf("カメラの初期化に失敗: %v", err)
	}

	width := webcam.Get(gocv.VideoCaptureFrameWidth)
	height := webcam.Get(gocv.VideoCaptureFrameHeight)

	detector, err := NewFaceDetector(width, height)
	if err != nil {
		webcam.Close()
		return nil, err
	}

	tmpl, err := template.ParseFiles("web/templates/index.html")
	if err != nil {
		webcam.Close()
		detector.Close()
		return nil, fmt.Errorf("テンプレートの読み込みに失敗: %v", err)
	}

	livenessDetector, err := NewLivenessDetector()
	if err != nil {
		webcam.Close()
		detector.Close()
		return nil, err
	}

	return &VideoHandler{
		webcam:           webcam,
		detector:         detector,
		template:         tmpl,
		frameSize:        image.Point{X: int(width), Y: int(height)},
		clients:          make(map[*websocket.Conn]bool),
		clientsMux:       sync.RWMutex{},
		livenessDetector: livenessDetector,
		done:             make(chan struct{}), // 追加
	}, nil
}

// Close リソースを解放
func (vh *VideoHandler) Close() {
	close(vh.done) // 終了を通知
	vh.clientsMux.Lock()
	for conn := range vh.clients {
		conn.Close()
	}
	vh.clients = make(map[*websocket.Conn]bool)
	vh.clientsMux.Unlock()

	if vh.webcam != nil {
		vh.webcam.Close()
	}
	if vh.detector != nil {
		vh.detector.Close()
	}
	if vh.livenessDetector != nil {
		vh.livenessDetector.Close()
	}
}

// ServeHTTP HTTPハンドラ
func (vh *VideoHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	vh.template.Execute(w, nil)
}

// HandleWebSocket WebSocket接続を処理
func (vh *VideoHandler) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket接続エラー: %v", err)
		return
	}

	vh.clientsMux.Lock()
	vh.clients[conn] = true
	vh.clientsMux.Unlock()

	defer func() {
		vh.clientsMux.Lock()
		delete(vh.clients, conn)
		vh.clientsMux.Unlock()
		conn.Close()
	}()

	// クライアント固有のチャネル
	clientDone := make(chan struct{})
	defer close(clientDone)

	// 顔検出用のゴルーチン
	go func() {
		detectImg := gocv.NewMat()
		defer detectImg.Close()

		ticker := time.NewTicker(frameDelay)
		defer ticker.Stop()

		for {
			select {
			case <-vh.done:
				return
			case <-clientDone:
				return
			case <-ticker.C:
				if vh.webcam == nil {
					return
				}

				vh.faceMutex.Lock()
				ok := vh.webcam.Read(&detectImg)
				if !ok || detectImg.Empty() {
					vh.faceMutex.Unlock()
					continue
				}

				faces := vh.detector.DetectFaces(&detectImg)
				vh.lastFaces = faces
				vh.faceMutex.Unlock()
			}
		}
	}()

	// 表示用のメインループ
	displayTicker := time.NewTicker(displayDelay)
	defer displayTicker.Stop()

	img := gocv.NewMat()
	defer img.Close()

	for {
		select {
		case <-vh.done:
			return
		case <-clientDone:
			return
		case <-displayTicker.C:
			if vh.webcam == nil {
				return
			}

			vh.faceMutex.Lock()
			ok := vh.webcam.Read(&img)
			if !ok || img.Empty() {
				vh.faceMutex.Unlock()
				continue
			}

			// 左右反転
			flipped := gocv.NewMat()
			gocv.Flip(img, &flipped, 1)
			vh.faceMutex.Unlock()

			vh.faceMutex.RLock()
			faces := vh.lastFaces
			vh.faceMutex.RUnlock()

			if err := vh.processFacesAndSend(&flipped, faces, conn); err != nil {
				log.Printf("画像送信エラー: %v", err)
				flipped.Close()
				return
			}
			flipped.Close()
		}
	}
}

// processFacesAndSend 顔検出結果を処理してWebSocketに送信
func (vh *VideoHandler) processFacesAndSend(img *gocv.Mat, faces []image.Rectangle, conn *websocket.Conn) error {
	width := img.Cols()

	for _, face := range faces {
		// 反転した画像に合わせて顔の位置を反転
		flippedFace := image.Rectangle{
			Min: image.Point{
				X: width - face.Max.X, // X座標を反転
				Y: face.Min.Y,         // Y座標はそのまま
			},
			Max: image.Point{
				X: width - face.Min.X, // X座標を反転
				Y: face.Max.Y,         // Y座標はそのまま
			},
		}

		faceRegion := img.Region(flippedFace)
		livenessResult := vh.livenessDetector.DetectLiveness(faceRegion)
		faceRegion.Close()

		borderColor := color.RGBA{0, 0, 255, 0} // blue
		if livenessResult.IsLive {
			borderColor = color.RGBA{0, 255, 0, 0} // green
		}

		gocv.Rectangle(img, flippedFace, borderColor, 3)
		text := fmt.Sprintf("%.1f%% %s", livenessResult.Confidence*100, livenessResult.Reason)
		pt := image.Point{X: flippedFace.Min.X, Y: flippedFace.Min.Y - 20}
		gocv.PutText(img, text, pt, gocv.FontHersheyPlain, 2.0, borderColor, 3)
	}

	buf, err := gocv.IMEncode(".jpg", *img)
	if err != nil {
		return err
	}
	defer buf.Close()

	return conn.WriteJSON(map[string]interface{}{
		"image":     base64.StdEncoding.EncodeToString(buf.GetBytes()),
		"faceCount": len(faces),
		"timestamp": time.Now().UnixNano(),
	})
}

// リロード通知を送信
func (vh *VideoHandler) NotifyReload() {
	vh.clientsMux.RLock()
	defer vh.clientsMux.RUnlock()

	for client := range vh.clients {
		client.WriteJSON(map[string]interface{}{
			"type": "reload",
		})
	}
}

// 値を0-1の範囲に正規化する関数
func normalizeScore(value, min, max float64) float64 {
	if value < min {
		return 0
	}
	if value > max {
		return 1
	}
	return (value - min) / (max - min)
}

// 目の開閉状態を判定する関数を追加
func (ld *LivenessDetector) isEyeClosed(eye gocv.Mat) bool {
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(eye, &gray, gocv.ColorBGRToGray)

	mean := gocv.NewMat()
	stddev := gocv.NewMat()
	defer mean.Close()
	defer stddev.Close()

	gocv.MeanStdDev(gray, &mean, &stddev)
	return mean.GetDoubleAt(0, 0) < 100
}

// main メイン関数
func main() {
	handler, err := NewVideoHandler()
	if err != nil {
		log.Fatal(err)
	}
	defer handler.Close()

	r := mux.NewRouter()
	r.HandleFunc("/", handler.ServeHTTP)
	r.HandleFunc("/ws", handler.HandleWebSocket)

	srv := &http.Server{
		Addr:    port,
		Handler: r,
	}

	// グレースフルシャットダウン用のチャネル
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("サーバーを起動しました: http://localhost%s", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	<-stop
	log.Println("シャットダウンを開始します...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("シャットダウンエラー: %v", err)
	}
	log.Println("サーバーを停止しました")
}
