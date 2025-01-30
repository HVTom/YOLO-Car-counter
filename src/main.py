import sys
import gi

gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
gi.require_version('Gdk', '3.0')

try:
    from gi.repository import Gtk, GdkPixbuf, Gdk
except ImportError as e:
    print(f"Import error: {e}")
    print("Paths searched:", gi.get_repository_search_path())
    print("Typelib path:", gi.get_typelib_path())
    sys.exit(1)

import cv2
import numpy as np

# incarcare model YOLO si class names
net = cv2.dnn.readNet('../config/yolov3.weights', '../config/yolov3.cfg')

with open('../config/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


class CarCounterApp(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Car Counter")
        self.set_border_width(10)
        self.set_default_size(800, 600)
        self.set_resizable(False)

        # Main vertical box
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(main_box)

        # Input Image Label
        input_label = Gtk.Label(label="Input Image")
        main_box.pack_start(input_label, False, False, 0)

        # Original Image Display Area
        self.original_image_display = Gtk.Image()
        main_box.pack_start(self.original_image_display, False, False, 0)

        # Processed Image Display Area
        processed_label = Gtk.Label(label="Processed Image")
        main_box.pack_start(processed_label, False, False, 0)

        self.processed_image_display = Gtk.Image()
        main_box.pack_start(self.processed_image_display, False, False, 0)

        # Select Image Button
        select_button = Gtk.Button(label="Select Image")
        select_button.connect("clicked", self.on_select_image)
        main_box.pack_start(select_button, False, False, 0)

        # Number of Cars Count
        self.count_label = Gtk.Label(label="Car Count: 0")
        main_box.pack_start(self.count_label, False, False, 0)

        # Class variables
        self.input_image_path = None
        self.extracted_image = None

    def on_select_image(self, widget):
        # file chooser dialog
        dialog = Gtk.FileChooserDialog(
            title="Select an Image",
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )

        # filter to accept all image file types
        filter_image = Gtk.FileFilter()
        filter_image.set_name("All image files")
        filter_image.add_mime_type("image/*")  # Accept all image file types
        dialog.add_filter(filter_image)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.input_image_path = dialog.get_filename()

            # Display the selected original image with fixed size
            self.display_fixed_size_image(self.input_image_path, self.original_image_display)

            # Process the image for object detection
            self.process_image(self.input_image_path)

        dialog.destroy()

    def display_fixed_size_image(self, image_path, image_widget):
        # Load the image and scale it to a fixed size (e.g., 400x300)
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
            image_path, 400, 300, True
        )
        image_widget.set_from_pixbuf(pixbuf)

    def process_image(self, input_file):
        if input_file:
            # Read and process the image with OpenCV
            frame = cv2.imread(input_file)
            height, width, _ = frame.shape

            # Prepare image for YOLO
            # conversion to a 4D blob for neural network
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers) # exit layers result

            class_ids = []
            confidences = []
            boxes = []

            # process YOLO outputs
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    #if confidence > 0.5:  # Confidence threshold
                    # calc object center
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    # bounding box dimensions
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    #top left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
            # eliminate overlapped detections
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Counting cars
            total_count = 0
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])

                    # Only count cars with confidence > 0.5
                    if label == 'car':
                        color = (255, 0, 0)  # rectangle color
                        # draw rectangle
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{label} {round(confidences[i], 2)}",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        total_count += 1

            # Update the car count label
            self.count_label.set_text(f"Car Count: {total_count}")

            # Display the result image with OpenCV (converted to RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_processed_image(frame_rgb)

    def display_processed_image(self, image):
        # Resize the image to a fixed size before converting to GdkPixbuf
        resized_image = cv2.resize(image, (400, 300), interpolation=cv2.INTER_AREA)

        # Convert the resized image to a GdkPixbuf
        height, width, _ = resized_image.shape
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            resized_image.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            width,
            height,
            width * 3
        )

        # Display the processed image in the GTK window
        self.processed_image_display.set_from_pixbuf(pixbuf)


def main():
    app = CarCounterApp()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
