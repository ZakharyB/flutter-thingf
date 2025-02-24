import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:html' as html;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  List<CameraDescription> cameras = [];
  try {
    if (kIsWeb) {
      await _requestWebCameraAccess();
    }
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Camera initialization error: $e');
  }
  runApp(MyApp(cameras: cameras));
}

Future<void> _requestWebCameraAccess() async {
  try {
    final stream = await html.window.navigator.mediaDevices?.getUserMedia({'video': true});
    if (stream != null) {
      stream.getTracks().forEach((track) => track.stop());
    }
  } catch (e) {
    debugPrint('Web camera access error: $e');
  }
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({Key? key, required this.cameras}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: CameraScreen(cameras: cameras),
    );
  }
}

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraScreen({Key? key, required this.cameras}) : super(key: key);

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  bool _isRearCameraSelected = true;
  bool _flashEnabled = false;
  double _minZoomLevel = 1.0;
  double _maxZoomLevel = 1.0;
  double _currentZoomLevel = 1.0;
  bool _isCameraInitialized = false;
  Timer? _cameraCheckTimer;
  String? _errorMessage;
  html.VideoElement? _webVideoElement;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    if (kIsWeb) {
      await _initializeWebCamera();
    } else {
      _initializeMobileCamera();
    }
    _startCameraCheck();
  }

  Future<void> _initializeWebCamera() async {
    try {
      final stream = await html.window.navigator.mediaDevices?.getUserMedia({'video': true});
      if (stream != null) {
        _webVideoElement = html.VideoElement()
          ..srcObject = stream
          ..autoplay = true;
        _isCameraInitialized = true;
        setState(() {});
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to access camera. Please check permissions.';
      });
    }
  }

  void _initializeMobileCamera() {
    if (widget.cameras.isNotEmpty && !_isCameraInitialized) {
      _initializeCameraController(widget.cameras[_isRearCameraSelected ? 0 : 1]);
    }
  }

  void _startCameraCheck() {
    _cameraCheckTimer = Timer.periodic(const Duration(seconds: 5), (timer) async {
      if (!_isCameraInitialized && mounted) {
        try {
          if (!kIsWeb) {
            final cameras = await availableCameras();
            if (cameras.isNotEmpty) {
              setState(() {
                widget.cameras.clear();
                widget.cameras.addAll(cameras);
              });
              _initializeMobileCamera();
            }
          }
        } catch (e) {
          debugPrint('Error checking cameras: $e');
        }
      }
    });
  }

  Future<void> _initializeCameraController(CameraDescription camera) async {
    if (_controller != null) {
      await _controller!.dispose();
    }

    try {
      _controller = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
      );

      _initializeControllerFuture = _controller?.initialize().then((_) async {
        if (!mounted) return;
        try {
          _minZoomLevel = await _controller!.getMinZoomLevel();
          _maxZoomLevel = await _controller!.getMaxZoomLevel();
          _isCameraInitialized = true;
          setState(() {});
        } catch (e) {
          debugPrint('Error getting zoom levels: $e');
          _isCameraInitialized = false;
        }
      }).catchError((error) {
        debugPrint('Camera initialization error: $error');
        _isCameraInitialized = false;
        setState(() {});
      });
    } catch (e) {
      debugPrint('Camera controller creation error: $e');
      _isCameraInitialized = false;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    } else if (state == AppLifecycleState.inactive) {
      _disposeCamera();
    }
  }

  void _disposeCamera() {
    _isCameraInitialized = false;
    if (kIsWeb) {
      _webVideoElement?.srcObject?.getTracks().forEach((track) => track.stop());
      _webVideoElement = null;
    } else {
      _controller?.dispose();
      _controller = null;
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _disposeCamera();
    _cameraCheckTimer?.cancel();
    super.dispose();
  }

  Future<void> _toggleCamera() async {
    if (kIsWeb) {
      return;
    }
    
    if (widget.cameras.length < 2) return;

    try {
      _isRearCameraSelected = !_isRearCameraSelected;
      await _initializeCameraController(
        widget.cameras[_isRearCameraSelected ? 0 : 1],
      );
    } catch (e) {
      debugPrint('Error toggling camera: $e');
    }
  }

  Future<void> _captureImage() async {
    if (kIsWeb) {
      return;
    }

    if (!_isCameraInitialized || _controller == null) return;

    try {
      await _initializeControllerFuture;
      final XFile? image = await _controller?.takePicture();
      if (image != null) {
        debugPrint('Image captured: ${image.path}');
      }
    } catch (e) {
      debugPrint('Error capturing image: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [
            _mainCameraPreview(),
            if (_isCameraInitialized) ...[
              _topControls(),
              _bottomControls(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _mainCameraPreview() {
    if (_errorMessage != null) {
      return Center(
        child: Text(
          _errorMessage!,
          style: const TextStyle(color: Colors.white, fontSize: 18),
          textAlign: TextAlign.center,
        ),
      );
    }

    if (kIsWeb) {
      if (_webVideoElement != null) {
        return HtmlElementView(
          viewType: 'webCamera',
          onPlatformViewCreated: (int id) {
          },
        );
      }
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }

    if (widget.cameras.isEmpty) {
      return const Center(
        child: Text(
          'Searching for cameras...',
          style: TextStyle(color: Colors.white, fontSize: 24),
        ),
      );
    }

    return FutureBuilder<void>(
      future: _initializeControllerFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done && _isCameraInitialized) {
          return Container(
            width: double.infinity,
            height: double.infinity,
            child: CameraPreview(_controller!),
          );
        }
        return const Center(
          child: CircularProgressIndicator(color: Colors.white),
        );
      },
    );
  }

  Widget _topControls() {
    return Positioned(
      top: 20,
      left: 0,
      right: 0,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          if (!kIsWeb) ...[
            IconButton(
              icon: Icon(
                _flashEnabled ? Icons.flash_on : Icons.flash_off,
                color: Colors.white,
                size: 28,
              ),
              onPressed: () {
                if (_controller != null && _isCameraInitialized) {
                  setState(() {
                    _flashEnabled = !_flashEnabled;
                    _controller!.setFlashMode(
                      _flashEnabled ? FlashMode.torch : FlashMode.off,
                    );
                  });
                }
              },
            ),
            Container(
              width: 200,
              child: Slider(
                value: _currentZoomLevel,
                min: _minZoomLevel,
                max: _maxZoomLevel,
                onChanged: (value) {
                  if (_controller != null && _isCameraInitialized) {
                    setState(() {
                      _currentZoomLevel = value;
                      _controller?.setZoomLevel(value);
                    });
                  }
                },
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _bottomControls() {
    return Positioned(
      bottom: 20,
      left: 0,
      right: 0,
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              IconButton(
                icon: const Icon(Icons.image, color: Colors.white, size: 28),
                onPressed: () {
                },
              ),
              GestureDetector(
                onTap: _captureImage,
                child: Container(
                  height: 60,
                  width: 60,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 3),
                  ),
                ),
              ),
              if (!kIsWeb)
                IconButton(
                  icon: const Icon(Icons.flip_camera_ios, color: Colors.white, size: 28),
                  onPressed: _toggleCamera,
                ),
            ],
          ),
        ],
      ),
    );
  }
}