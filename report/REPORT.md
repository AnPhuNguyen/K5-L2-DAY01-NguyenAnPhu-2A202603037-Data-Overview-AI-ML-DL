# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Python

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
**Trả lời**
```json
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  },
```

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
**Trả lời:** Record / bản kết quả này nói về một vật thể riêng trong ảnh, cụ thể là cab. lý do: đối với "sample_id": "traffic" nói riêng, có các khóa / key với giá trị / value khác nhau: "rank", "class_id", "class_name", "score"

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
**Trả lời:**  Tự đoán và đi kèm với checkpoint yolo11n-cls.pt

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
**Trả lời:** 
- ghi thêm class_id đề nhận diện record dễ hơn, nhất là khi cần viết code ra báo cáo liên quan record
- ghi thêm taxonomy_name: để chỉ đến cơ sở dữ liệu / database hình ảnh để dựa vào đó mà tiến hành nhận diện
- ghi thêm class_name để định nghĩa thêm record đó có tên lớp nào, theo cách con người có thể hiểu được

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
**Trả lời:**  guideline cần quy định rõ cần ưu tiên vật thể như thế nào. Cụ thể:
- Chọn vật nổi bật nhất: to nhất, gần trung tâm ảnh nhất hay độ phân giải cao nhất, rõ nhất. Cần ràng buộc thêm nếu ban đầu không tìm ra vật nổi bật
- Bỏ qua các vật thể nền, phụ trợ hoặc bị che khuất
- Chọn vật mang ý nghĩa nhiều nhất trong ngữ cảnh của bức ảnh

6. Vì sao không được lấy `score` của model làm ground truth?
**Trả lời:**  có nguy cơ tạo ra vòng lặp xác nhận, làm mô hình chỉ có thể phán đoán đúng dựa trên những vật thể sẵn có thay vì các trướng hợp đa dạng hơn trong thực tế

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
**Trả lời:**
```json
  {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 45,
    "class_name": "bowl",
    "score": 0.701282,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      155.02,
      168.67,
      182.38,
      184.18
    ],
    "bbox_width": 27.36,
    "bbox_height": 15.51
  }
```
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
**Trả lời:**
- Vị trí toạn độ của góc trên-trái của bounding box: (155.02, 168.67) 
- Vị trí toạn độ của góc dưới-phải của bounding box: (182.38, 184.18) 
- bounding box rộng 27.36 pixel, cao 15.31 pixel

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
**Trả lời:** Lấy 2 ngưỡng 0.20 và 0.60
- Tại ngưỡng 0.20: Phát hiện được 18 vật thể.

  - Danh sách lớp dự đoán: ['person', 'oven', 'sink', 'bowl', 'refrigerator', 'potted plant', 'cup', 'knife', 'knife', 'knife', 'bowl', 'bottle', 'cup', 'spoon', 'bottle', 'dining table', 'bowl', 'knife']

- Tại ngưỡng 0.60: Phát hiện được 3 vật thể.

  - Danh sách lớp dự đoán: ['person', 'oven', 'refrigerator']

(lấy prediction record với score >= ngưỡng confidence đề ra)

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
**Trả lời:** người review (kiểm duyệt) phải thực hiện chọn lọc thủ công các kết quả thật sự đúng, chính xác và loại bỏ các kết quả sai, kết quả nhận diện giả.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
**Trả lời:** 
Xác định loại nhãn phù hợp: Áp dụng Nhãn phân loại khi cần gán một thẻ duy nhất cho toàn bộ hình ảnh; sử dụng Hộp giới hạn (Bounding Box) để định vị và phân loại các đối tượng riêng biệt; dùng Đa giác (Polygon) hoặc Mặt nạ phân đoạn (Segmentation Masks) để đánh dấu chính xác ranh giới của đối tượng hoặc phân loại từng pixel.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
**Trả lời:** vật bị che khuất: "person 0.61". Thêm quy tắc: không nhận diện vật nếu không thể nhìn thấy toàn thể, hoặc giảm confidence xuống. Lý do: chỉ nhìn thấy tay nhưng gán nhãn "person" mà "confidence" 0.61 là cao một cách vô lý

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
**Trả lời:**
  {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-001",
    "class_id": 0,
    "class_name": "person",
    "score": 0.899318,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.45,
      66.44,
      498.02,
      348.58
    ],
    "polygon_point_count": 348,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ]
      
    ]
  }


2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
**Trả lời:**
Hình dạng thực tế (bám biên): Bounding box (bbox) chỉ giới hạn vật thể trong một khung hình chữ nhật cố định, dẫn đến việc chứa nhiều pixel của nền hoặc của vật thể lân cận.
Polygon bám sát đường biên thực tế (contour) của vật thể. Nó cho biết chính xác vùng không gian/hình học chi tiết của đối tượng (ví dụ: tư thế người, hình dáng bo tròn của cái bát, độ thon dài của con dao) và xác định đúng pixel nào thực sự thuộc về đối tượng đó.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
**Trả lời:**
Mục đích: instance_id dùng để phân biệt và định danh riêng biệt từng cá thể đơn lẻ xuất hiện trong ảnh tĩnh.
Cách hiểu câu "Không phải class ID và không phải tracking ID":
Không phải class ID: Class ID là mã định danh chung cho cả nhóm loài/nhãn (ví dụ: tất cả con người đều chung class_id = 0). Còn instance_id sẽ phân biệt người thứ nhất (kitchen-001) với người thứ hai (kitchen-002).
Không phải tracking ID: Tracking ID là ID dùng để bám đuổi (track) một vật thể xuyên suốt qua nhiều khung hình khác nhau của một đoạn video. Ở đây, instance_id chỉ có tính chất cục bộ, độc lập và chỉ có giá trị trên duy nhất một bức ảnh tĩnh này.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
**Trả lời:**
Quy tắc gán nhãn đường biên (Polygon Annotation Rule): "Khi vẽ đa giác (polygon), người gán nhãn phải đi sát theo đường viền thực tế của vật thể. Trong trường hợp vật thể bị che khuất một phần (occlusion), chỉ vẽ phần nhìn thấy rõ (visible part), tuyệt đối không được tự ý vẽ lấn sang vùng của vật thể khác ở phía trước hoặc đoán mò đường bao của phần bị che mất."

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
**Trả lời:**
Điểm khó tiêu biểu: Khu vực các con dao (knife) và chiếc bát (bowl) nằm rất sát nhau trên bàn bếp, hoặc vùng tối nhập nhằng giữa chân người đầu bếp và lò nướng (oven).
Lựa chọn cách xử lý: Ưu tiên dùng Guideline trước, nếu không có quy định sẵn thì mới dùng Escalation.
Lý do: Dự án gán nhãn quy mô lớn luôn cần tính nhất quán cực kỳ cao. Nếu trong Guideline đã có sẵn quy tắc xử lý trường hợp chạm nhau (ví dụ: 'Nếu hai vật chạm nhau, vẽ đường biên bám sát theo ranh giới màu sắc phân tách'), annotator chỉ cần làm theo. Nhưng nếu ranh giới đó hoàn toàn bị mờ đen, không thể nhìn thấy bằng mắt thường, annotator bắt buộc phải Escalation (báo cáo lên QC/PM) để nhận hướng dẫn xử lý đồng bộ, tránh việc tự đoán mò làm giảm chất lượng dữ liệu huấn luyện.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
**Trả lời:**
Quy tắc bảo vệ dữ liệu cho bài lab này là chỉ thu thập dữ liệu cần thiết cho huấn luyện trừ dữ liệu cá nhân, chỉ cho người có thẩm quyền hay được phép mới có thể truy cập dữ liệu, dùng các phương thức mã hóa để phòng tránh lộ dữ liệu, hành vi bất thường hay trích xuất mô hình trái phép.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
**Trả lời:** cần báo cáo đến người hướng dẫn hay người giám sát có trách nhiệm với mô hình ở cấp cao, hoặc người vận hành mô hình

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
