import sys, os, argparse
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from models.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', type=str,
                        default='./pretrained_models/double_backbone_1022.pth')
    parser.add_argument('--input_dir', type=str, default='./demo/imgs', help='input data dir')
    parser.add_argument('--output_dir', type=str, default='./demo/res', help='output dir')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    return args 




def append_mask(mask_img, img, body_color, mask_part=None, alpha=0.3, color_part = None):
	append_area = mask_img > 0.7
	mask_img[append_area] = 1
	mask_to_drawborder = np.zeros(mask_img.shape, dtype=np.uint8)
	mask_to_drawborder[append_area] = 255
	color = body_color
	if mask_part is not None:
		append_area_part = mask_part > 0.7
		mask_part[append_area_part] = 1
		mask_to_drawborder[append_area_part] = 255
		mask_img[append_area_part] = 0
		for i in range(3):
			img[:, :, i][append_area] = color[i] * mask_img[append_area] * alpha + img[:, :, i][append_area] * (1-alpha)
			img[:, :, i][append_area_part] = img[:, :, i][append_area_part] * (1-0.4) + color_part[i] * mask_part[append_area_part] * 0.4
		
		_, contours, _ = cv2.findContours(mask_to_drawborder.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(img, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)

	else:
		for i in range(3):
			img[:, :, i][append_area] = color[i] * mask_img[append_area] * alpha + img[:, :, i][append_area] * (1-alpha)
	
		_, contours, _ = cv2.findContours(mask_to_drawborder.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(img, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
	return img

def vis_class(img, pos, class_str, font_scale=1):
    """Visualizes the class."""
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, (255,0,0), -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, (18, 127, 15), 2)
    return img

def remove(normal_box, normal_mask, semantic_mask):
	new_normal_box = []
	new_normal_mask = []
	for n_mask, n_box in zip(normal_mask, normal_box):
		is_valid = True
		for s_mask in semantic_mask:
			if caculate_iou(n_mask, s_mask) > 0.6:
				is_valid = False
		if is_valid:
			new_normal_box.append(n_box)
			new_normal_mask.append(n_mask)
	
	return new_normal_box, new_normal_mask

def caculate_iou(mask1, mask2):
	mask1_binary = np.zeros(mask1.shape)
	mask2_binary = np.zeros(mask2.shape)
	mask1_binary[mask1 > 0.5] = 1
	mask2_binary[mask2 > 0.5] = 1
	mask1_binary = mask1_binary == 1
	mask2_binary = mask2_binary == 1
	intersection = (mask1_binary & mask2_binary)
	union = (mask1_binary | mask2_binary)
	iou = np.sum(intersection) / np.sum(union)
	return iou

def forword_net(model, img, save_path, states_descriptions, args):
	
	outputs = model([F.to_tensor(img).to(args.device)])
	outputs = [{k: v for k, v in t.items()} for t in outputs]
	labels = outputs[0]['labels'].cpu().numpy()
	bboxes = outputs[0]['boxes'].detach().cpu().numpy()
	scores = outputs[0]['scores'].detach().cpu().numpy()
	states = outputs[0]['states'].detach().cpu().numpy()
	masks = outputs[0]['masks'].detach().cpu().numpy()
	partmasks = outputs[0]["partmasks"].detach().cpu().numpy()
	img_np = np.array(img)
	boxes_to_draw = []
	states_des_to_show = []
	scores_to_show = []
	
	semantic_mask = []
	normal_mask = []
	semantic_part = []
	semantic_box = []
	normal_box = []
	for i in range(labels.shape[0]):
		if scores[i] > 0.8:
			bbox = bboxes[i]
			scores_to_show.append(scores[i])
			boxes_to_draw.append(bbox)
			state_des = ""
			
			if labels[i] == 2:
				state = states[i, :]
				index = np.argmax(state)
				state_des = states_descriptions[index]
				states_des_to_show.append(state_des)
				mask_img = masks[i][0]
				mask_part = partmasks[i][0]
				semantic_mask.append(mask_img)
				semantic_part.append(mask_part)
				semantic_box.append(bbox)
				# img_np = append_mask(mask_img, img_np, [109, 212, 0], mask_part=mask_part)
				# img_np = append_heat_map(mask_part, img_np)
			else:
				
				# states_des_to_show.append("normal")
				mask_img = masks[i][0]
				normal_mask.append(mask_img)
				normal_box.append(bbox)
				# img_np = append_mask(mask_img, img_np, [132, 125, 2], alpha=0.4)

	new_normal_box, new_normal_mask = remove(normal_box, normal_mask, semantic_mask)

	for mask_img in new_normal_mask:
		img_np = append_mask(mask_img, img_np, [255, 255, 0], alpha=0.3)
	for mask_img, mask_part in zip(semantic_mask, semantic_part):
		img_np = append_mask(mask_img, img_np, [255, 80, 0], mask_part=mask_part, color_part=[0, 145, 255])
	for states, box in zip(states_des_to_show, semantic_box):
		img_np = vis_class(img_np, box, states, font_scale=2)
		
	img = Image.fromarray(img_np.astype('uint8')).convert('RGB')
	draw = ImageDraw.Draw(img)
	box_width = 1
	font = ImageFont.truetype('FreeMono.ttf', 20)
	if img_np.shape[1] == 3384:
		box_width = 3
		font = ImageFont.truetype('FreeMono.ttf', 40)
	elif img_np.shape[1] == 1920 or img_np.shape[1] == 2048:
		box_width = 2
		font = ImageFont.truetype('FreeMono.ttf', 30)


	for bbox in (new_normal_box):
		draw.line((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[0]), int(bbox[3]), int(bbox[0]), int(bbox[1])), 'green', box_width)
	for bbox in (semantic_box):
		draw.line((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[0]), int(bbox[3]), int(bbox[0]), int(bbox[1])), 'red', box_width)
		
	
	img.save(save_path)

def infer_sample(args):
	states_descriptions = ["Bonnet is lifted", "Trunk is lifted", "Front-left door is opened", "Front-right door is opended", "Back-left door is opened", "Back-right door is opened"]

	state_dict = torch.load(args.pretrained_model)

	model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=3, is_double_backbone=True)	

	model.load_state_dict(state_dict['model'])
	model.to(args.device)
	model.eval()
	imgdir = args.input_dir
	
	
	imgs = [f for f in os.listdir(imgdir)]
	for i, img_name in enumerate(imgs):
		path = os.path.join(imgdir, img_name)
		img = Image.open(path).convert("RGB")
		save_path = os.path.join(args.output_dir, img_name)

		forword_net(model, img, save_path, states_descriptions, args)

		print(str(i) + '/' + str(len(imgs)))

def main():
    args = parse_args()
    infer_sample(args)


if __name__ == '__main__':
    main()