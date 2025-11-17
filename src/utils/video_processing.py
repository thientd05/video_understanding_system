from decord import VideoReader, cpu, gpu
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector




def video_processing(video_path: str) -> list:
    ans = []    
    #scene detect and collect frames
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.start()
    scene_manager.detect_scenes(video_manager)
    scene_list = scene_manager.get_scene_list()
    print(f"Số cảnh: {len(scene_list)}")
    frames_to_sample = [(start.get_frames() + end.get_frames()) // 2 for start, end in scene_list]
    
    try:
        gpu = gpu(0)
        vr = VideoReader(video_path, ctx=gpu)
    except Exception:
        print("using cpu")
        vr = VideoReader(video_path, ctx=cpu())
        
    
    for frame in frames_to_sample:
        image = vr[frame].asnumpy()
        ans.append(image)
        
    return ans

        