from ultralytics import YOLO
from PIL import Image
import os.path


path = 'pond_turtle.jpg' #filename to image you want to use model on
check_file = os.path.isfile(path)
if not check_file:
  print("image file does not exist")
  exit()

model = YOLO("turtle_bestv8s.pt")
results = model.predict(path,conf = 0.65)
result = results[0]
#print(len(result.boxes)) #prints how many objects detected
#print(result.names) #prints all the class names

for box in result.boxes:
  class_id = result.names[box.cls[0].item()]
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  conf = round(box.conf[0].item(), 2)
  print("Object type:", class_id)
  print("Coordinates:", cords)
  print("Probability:", conf)
  print("---")


im = Image.fromarray(result.plot()[:,:,::-1])
im.show()
im.save("output-"+path)