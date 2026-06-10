import xml.etree.ElementTree as ET
import uuid

# --- Visual Styles ---
STYLES = {
    # Step boxes
    "step_input": "rounded=1;whiteSpace=wrap;html=1;fillColor=#C5FBFF;strokeColor=#0e8088;fontSize=11;fontFamily=Helvetica;align=left;verticalAlign=top;spacingLeft=10;spacingTop=5;strokeWidth=2;",
    "step_encoder": "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;fontFamily=Helvetica;align=left;verticalAlign=top;spacingLeft=10;spacingTop=5;strokeWidth=2;",
    "step_decoder": "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;fontFamily=Helvetica;align=left;verticalAlign=top;spacingLeft=10;spacingTop=5;strokeWidth=2;",
    "step_loss": "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=11;fontFamily=Helvetica;align=left;verticalAlign=top;spacingLeft=10;spacingTop=5;strokeWidth=2;",
    "step_output": "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=11;fontFamily=Helvetica;align=left;verticalAlign=top;spacingLeft=10;spacingTop=5;strokeWidth=2;",
    
    # Arrows
    "arrow_main": "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=3;strokeColor=#333333;endArrow=block;endFill=1;",
    "arrow_skip": "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#d79b00;dashed=1;endArrow=block;endFill=1;",
    
    # Title
    "text_title": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=18;fontStyle=1;fontFamily=Helvetica;",
    "text_step_num": "ellipse;whiteSpace=wrap;html=1;fillColor=#333333;strokeColor=#333333;fontSize=14;fontStyle=1;fontColor=#ffffff;",
}

OUTPUT_FILE = r"g:\model_training\sami_sir\phase_1\lab_pc_seg\kidney\output\meddino_vista3d_v4_tumor_focused\meddino_v4_method.drawio.xml"

def create_id():
    return "_" + str(uuid.uuid4()).replace("-", "")[:20]

def create_cell(root, id, value, style, vertex=False, edge=False, source=None, target=None, parent="1", geometry=None):
    cell = ET.SubElement(root, "mxCell")
    cell.set("id", id)
    if value:
        cell.set("value", value)
    if style:
        cell.set("style", style)
    if vertex:
        cell.set("vertex", "1")
    if edge:
        cell.set("edge", "1")
    if source:
        cell.set("source", source)
    if target:
        cell.set("target", target)
    cell.set("parent", parent)
    
    if geometry:
        geo = ET.SubElement(cell, "mxGeometry")
        if vertex:
            geo.set("x", str(geometry.get("x", 0)))
            geo.set("y", str(geometry.get("y", 0)))
            geo.set("width", str(geometry.get("width", 100)))
            geo.set("height", str(geometry.get("height", 50)))
        if edge:
            geo.set("relative", "1")
        geo.set("as", "geometry")
    return cell

def create_step_box(root, step_num, title, description, x, y, width, height, style):
    """Create a step box with number, title, and description"""
    box_id = create_id()
    
    # Format the content
    content = f"&lt;b&gt;STEP {step_num}: {title}&lt;/b&gt;&lt;br&gt;&lt;br&gt;{description}"
    
    create_cell(root, box_id, content, style, vertex=True,
                geometry={"x": x, "y": y, "width": width, "height": height})
    
    return box_id

def generate_diagram():
    # Create root structure
    root_root = ET.Element("mxfile")
    root_root.set("host", "app.diagrams.net")
    root_root.set("agent", "Mozilla/5.0")
    root_root.set("version", "29.3.6")

    diagram = ET.SubElement(root_root, "diagram")
    diagram.set("id", "MedDINO_V4_Flow")
    diagram.set("name", "Page-1")

    mxGraphModel = ET.SubElement(diagram, "mxGraphModel")
    mxGraphModel.set("dx", "1489")
    mxGraphModel.set("dy", "1200")
    mxGraphModel.set("grid", "0")
    mxGraphModel.set("gridSize", "10")
    mxGraphModel.set("guides", "1")
    mxGraphModel.set("tooltips", "1")
    mxGraphModel.set("connect", "1")
    mxGraphModel.set("arrows", "1")
    mxGraphModel.set("fold", "1")
    mxGraphModel.set("page", "1")
    mxGraphModel.set("pageScale", "1")
    mxGraphModel.set("pageWidth", "850")
    mxGraphModel.set("pageHeight", "1400")
    mxGraphModel.set("math", "0")
    mxGraphModel.set("shadow", "0")

    root = ET.SubElement(mxGraphModel, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    # Title
    title_id = create_id()
    create_cell(root, title_id, "&lt;b&gt;MedDINO-VISTA3D V4: Step-by-Step Process Flow&lt;/b&gt;", 
                STYLES["text_title"], vertex=True, 
                geometry={"x": 50, "y": 20, "width": 750, "height": 30})

    # Layout parameters
    box_width = 700
    box_height = 100
    start_x = 75
    start_y = 70
    spacing_y = 20
    
    current_y = start_y
    step_ids = []

    # STEP 1: Input
    step1_id = create_step_box(
        root, 1, "Load CT Scan",
        "Take a 3D CT scan of the kidney (Patch Size: 140×224×224 voxels). This is the input volume for the model.",
        start_x, current_y, box_width, box_height, STYLES["step_input"]
    )
    step_ids.append(step1_id)
    current_y += box_height + spacing_y

    # STEP 2: Balanced Patch Sampling
    step2_id = create_step_box(
        root, 2, "Balanced Patch Sampling",
        "Sample 50% of patches from tumor regions, 25% from cyst, 15% from kidney, and 10% from random areas. Use `np.argwhere` to find target classes and extract 140×224×224 volumes around random centroids.",
        start_x, current_y, box_width, box_height, STYLES["step_input"]
    )

    step_ids.append(step2_id)
    current_y += box_height + spacing_y

    # STEP 3: Slice Conversion
    step3_id = create_step_box(
        root, 3, "Convert to 2D Slices",
        "The model is a hybrid 2D-3D architecture. The 3D patch (140×224×224) is flattened into 140 separate 2D slices of 224×224 pixels. Each slice is processed independently by DINOv2.",
        start_x, current_y, box_width, box_height, STYLES["step_encoder"]
    )
    step_ids.append(step3_id)
    current_y += box_height + spacing_y

    # STEP 4: DINOv2 Feature Extraction
    step4_id = create_step_box(
        root, 4, "Extract Features with DINOv2",
        "Pass the 140 slices through the frozen DINOv2 backbone. It sees 140 'images'. It breaks each 224×224 slice into 14×14 patches and extracts feature maps.",
        start_x, current_y, box_width, box_height, STYLES["step_encoder"]
    )
    step_ids.append(step4_id)
    current_y += box_height + spacing_y

    # STEP 5: Multi-Scale Features
    step5_id = create_step_box(
        root, 5, "Collect Multi-Scale Features",
        "Grab features from DINOv2 layers 2, 5, 8, and 11 for each slice. We now have 140 sets of feature maps."
        , start_x, current_y, box_width, box_height, STYLES["step_encoder"]
    )
    step_ids.append(step5_id)
    current_y += box_height + spacing_y

    # STEP 6: 3D Reconstruction & Fusion
    step6_id = create_step_box(
        root, 6, "Reconstruct 3D Volume",
        "Stack the 2D feature maps back together to form a 3D volume (140×16×16). A 'Depth Aggregator' (3D Convolution) then mixes information across slices to understand 3D structure.",
        start_x, current_y, box_width, box_height, STYLES["step_encoder"]
    )
    step_ids.append(step6_id)
    current_y += box_height + spacing_y

    # STEP 7: First Upsampling
    step7_id = create_step_box(
        root, 7, "Upsample and Add Skip Connection #1",
        "Upsample features by 2x (Depth, Height, Width). Add back the saved features from layer 11 (interpolated to match). Use spatial attention to focus on important regions.",
        start_x, current_y, box_width, box_height, STYLES["step_decoder"]
    )
    step_ids.append(step7_id)
    current_y += box_height + spacing_y

    # STEP 8: Second Upsampling
    step8_id = create_step_box(
        root, 8, "Upsample and Add Skip Connection #2",
        "Upsample features by 2x again. Add back saved features from layer 8 (interpolated to match). Again use spatial attention to highlight important areas.",
        start_x, current_y, box_width, box_height, STYLES["step_decoder"]
    )
    step_ids.append(step8_id)
    current_y += box_height + spacing_y

    # STEP 9: Final Prediction & Interpolation
    step9_id = create_step_box(
        root, 9, "Prediction & Interpolation",
        "Generate raw predictions and interpolate (resize) to 140×224×224 to match the input volume and resolve upsampling mismatches.",
        start_x, current_y, box_width, box_height, STYLES["step_decoder"]
    )


    step_ids.append(step9_id)
    current_y += box_height + spacing_y

    # STEP 10: Loss Calculation
    step10_id = create_step_box(
        root, 10, "Calculate Loss",
        "Compare predictions to ground truth labels using three loss functions: Tversky Loss (50%, handles class imbalance), Focal Loss (30%, focuses on hard examples), and Dice Loss (20%, measures overlap). Heavily weight tumor and cyst classes.",
        start_x, current_y, box_width, box_height + 20, STYLES["step_loss"]
    )
    step_ids.append(step10_id)
    current_y += box_height + 20 + spacing_y

    # STEP 11: Backpropagation
    step11_id = create_step_box(
        root, 11, "Update Model Weights",
        "Use the loss to update the model's learnable parameters via backpropagation. The AdamW optimizer adjusts weights to minimize the loss. Only the decoder and fusion layers are updated (DINOv2 stays frozen).",
        start_x, current_y, box_width, box_height, STYLES["step_output"]
    )
    step_ids.append(step11_id)
    current_y += box_height + spacing_y

    # STEP 12: Repeat Training Loop
    step12_id = create_step_box(
        root, 12, "Iterative Training",
        "Repeat for 200 total epochs (divided into 2 stages). The loop stops when all epochs are finished OR when 'Early Stopping' triggers if the validation loss hasn't improved for 10 consecutive epochs.",
        start_x, current_y, box_width, box_height, STYLES["step_output"]
    )

    step_ids.append(step12_id)

    # Connect all steps with arrows
    for i in range(len(step_ids) - 1):
        edge_id = create_id()
        create_cell(root, edge_id, "", STYLES["arrow_main"], edge=True, 
                    source=step_ids[i], target=step_ids[i+1])

    # Add skip connection annotations (visual only, no actual arrows to avoid clutter)
    skip_note_y = start_y + (box_height + spacing_y) * 4 + 50
    skip_note_id = create_id()
    create_cell(root, skip_note_id, 
                "&lt;i&gt;Note: Skip connections carry detailed features from Steps 5→7 and 5→8&lt;/i&gt;",
                "text;html=1;strokeColor=#d79b00;fillColor=#fff2cc;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=1;fontSize=10;fontFamily=Helvetica;dashed=1;",
                vertex=True,
                geometry={"x": start_x + 150, "y": skip_note_y, "width": 400, "height": 30})

    # Save XML
    tree = ET.ElementTree(root_root)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT_FILE, encoding="UTF-8", xml_declaration=True)
    
    print(f"✓ Successfully generated step-by-step flow diagram!")
    print(f"  File: {OUTPUT_FILE}")
    print(f"  Steps: 12 sequential steps from input to training")
    print(f"  Layout: Top-to-bottom flow")
    print(f"  Style: Plain English descriptions, no complex equations")

if __name__ == "__main__":
    generate_diagram()
