"""
generate_ai_miniproject_ppt.py
Generates a simplified, friendly 16:9 widescreen PowerPoint presentation for the AI Mini-Project.
Features:
- Minimal math / formulas, replaced with intuitive plain English and analogies.
- Slide 9 (old SMOTE theory page) removed completely (11 slides total).
- Clean, approachable visual cards and native tables.
- Conversational embedded speaker notes on every slide.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Color Palette
    C_NAVY_DARK   = RGBColor(15, 23, 42)      # Deep Slate Navy
    C_BLUE_LIGHT  = RGBColor(2, 132, 199)     # Bright Blue
    C_BLUE_ACCENT = RGBColor(14, 116, 144)    # Teal / Cyan
    C_GREEN_ACC   = RGBColor(16, 185, 129)    # Emerald Green
    C_BG_LIGHT    = RGBColor(248, 250, 252)   # Soft Off-White
    C_CARD_BG     = RGBColor(255, 255, 255)   # Crisp White
    C_TEXT_MAIN   = RGBColor(30, 41, 59)      # Charcoal Text
    C_TEXT_MUTED  = RGBColor(100, 116, 139)   # Slate Muted Text
    C_BORDER      = RGBColor(226, 232, 240)   # Soft Border

    def set_bg(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category="ARTIFICIAL INTELLIGENCE MINI-PROJECT"):
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        p_cat = cat_box.text_frame.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_BLUE_LIGHT

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.7))
        p_t = title_box.text_frame.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = C_NAVY_DARK

    def add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.5)
        else:
            card.line.fill.background()
        return card

    def add_notes(slide, notes_text):
        tf = slide.notes_slide.notes_text_frame
        tf.text = notes_text

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1, C_NAVY_DARK)

    # Accent bar
    dec = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.4), Inches(1.2), Inches(0.08))
    dec.fill.solid()
    dec.fill.fore_color.rgb = C_GREEN_ACC
    dec.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.7), Inches(11.0), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "ARTIFICIAL INTELLIGENCE MINI-PROJECT"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = C_GREEN_ACC
    p0.space_after = Pt(14)

    p1 = tf.add_paragraph()
    p1.text = "Brain-Inspired AI for Real-Time\nHeart Rhythm Monitoring"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)
    p1.space_after = Pt(16)

    p2 = tf.add_paragraph()
    p2.text = "Accurate, Low-Power Arrhythmia Classification using Spiking Neural Networks (SNNs)"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(148, 163, 184)
    p2.space_after = Pt(36)

    p3 = tf.add_paragraph()
    p3.text = "Author: Ketan Shinde  |  Domain: Neuromorphic AI & Healthcare Machine Learning"
    p3.font.size = Pt(14)
    p3.font.bold = True
    p3.font.color.rgb = RGBColor(255, 255, 255)

    add_notes(s1,
        "Good morning everyone. Today I am presenting our AI mini-project: a brain-inspired Spiking Neural Network "
        "designed for real-time heart rhythm monitoring. Instead of using heavy, power-hungry deep learning models, "
        "we use neuromorphic artificial intelligence that communicates through biological pulses to detect cardiac arrhythmias "
        "with over 97% accuracy while using very little computing power."
    )

    # =========================================================================
    # SLIDE 2: Problem Statement & Clinical Motivation
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2, C_BG_LIGHT)
    add_header(s2, "Problem Statement & Clinical Motivation")

    cards_data = [
        ("The Medical Need", [
            "Heart diseases are the leading cause of mortality worldwide.",
            "Irregular heart rhythms (arrhythmias) can happen randomly and without warning.",
            "Patients need continuous 24/7 monitoring through small wearable patches.",
            "Catching dangerous heartbeats early prevents sudden cardiac arrest."
        ], C_BLUE_LIGHT),
        ("The Problem with Traditional AI", [
            "Standard deep learning models (like CNNs or LSTMs) do massive continuous calculations.",
            "They consume heavy battery power, draining wearable monitors in just a few hours.",
            "Sending raw heart signals to the cloud introduces lag and risks patient privacy.",
            "We need an AI that can think locally on the wearable device in real time."
        ], C_NAVY_DARK)
    ]

    for i, (ctitle, cbullets, tcol) in enumerate(cards_data):
        cx = Inches(0.8 + i * 6.0)
        add_card(s2, cx, Inches(1.6), Inches(5.7), Inches(5.2))
        tb = s2.shapes.add_textbox(cx + Inches(0.3), Inches(1.85), Inches(5.1), Inches(4.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = ctitle
        p_t.font.size = Pt(18)
        p_t.font.bold = True
        p_t.font.color.rgb = tcol
        p_t.space_after = Pt(16)

        for b in cbullets:
            pb = tf.add_paragraph()
            pb.text = "• " + b
            pb.font.size = Pt(14)
            pb.font.color.rgb = C_TEXT_MAIN
            pb.space_after = Pt(14)

    add_notes(s2,
        "To introduce our motivation: heart disease is the number one cause of death globally. "
        "Dangerous heart conditions often strike suddenly, so patients need wearable heart monitors they can wear all day and night. "
        "However, standard AI models are too heavy for small battery devices—they drain batteries in hours and relying on cloud servers "
        "causes delays when a patient is in danger. Our goal was to create an intelligent model light enough to run right on the device."
    )

    # =========================================================================
    # SLIDE 3: Basic Idea of SNNs
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3, C_BG_LIGHT)
    add_header(s3, "Understanding Spiking Neural Networks (SNNs)")

    snn_concepts = [
        ("1. Brain-Inspired Pulses", [
            "In our brains, neurons communicate using tiny electrical pulses called spikes.",
            "Instead of sending heavy decimal numbers, SNN neurons send simple on/off signals (0 or 1).",
            "Information is carried in the rhythm and timing of when these pulses fire."
        ], C_BLUE_LIGHT),
        ("2. The Leaky Bucket Analogy", [
            "Think of each artificial neuron like a bucket with a small leak at the bottom.",
            "Incoming pulses add water into the bucket (Integrate).",
            "If nothing arrives, the level slowly drops over time (Leak).",
            "When the bucket fills to the top, it fires a pulse and resets to empty (Fire & Reset)."
        ], C_BLUE_ACCENT),
        ("3. Why It Saves Power", [
            "Neurons stay completely silent until an actual pulse arrives.",
            "No spikes = no activity = zero power consumed.",
            "Instead of complex multiplications, the network just does simple additions when a pulse hits.",
            "Ideal for smart, battery-powered medical wearables."
        ], C_GREEN_ACC)
    ]

    for i, (ctitle, cbullets, col) in enumerate(snn_concepts):
        cx = Inches(0.8 + i * 3.95)
        add_card(s3, cx, Inches(1.6), Inches(3.8), Inches(5.2), border_color=col)
        tb = s3.shapes.add_textbox(cx + Inches(0.25), Inches(1.85), Inches(3.3), Inches(4.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = ctitle
        p_t.font.size = Pt(17)
        p_t.font.bold = True
        p_t.font.color.rgb = col
        p_t.space_after = Pt(16)

        for b in cbullets:
            pb = tf.add_paragraph()
            pb.text = "• " + b
            pb.font.size = Pt(13)
            pb.font.color.rgb = C_TEXT_MAIN
            pb.space_after = Pt(12)

    add_notes(s3,
        "Here is the simple concept behind Spiking Neural Networks. In the human brain, neurons don't exchange large decimal numbers; "
        "they fire discrete electrical pulses. SNNs mimic this exact behavior. Each artificial neuron acts like a bucket with a tiny leak: "
        "it collects incoming signals, leaks if nothing arrives, and when it reaches a set threshold, it fires a pulse and resets to zero. "
        "Because neurons stay completely quiet when there is nothing to report, SNNs use almost zero power during quiet intervals."
    )

    # =========================================================================
    # SLIDE 4: Dataset & The 4 Arrhythmia Classes
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4, C_BG_LIGHT)
    add_header(s4, "The Dataset: MIT-BIH Arrhythmia Benchmark")

    add_card(s4, Inches(0.8), Inches(1.5), Inches(11.7), Inches(1.2))
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(1.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Benchmark: MIT-BIH Arrhythmia Database (PhysioNet)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK
    p2 = tf.add_paragraph()
    p2.text = "Standard clinical benchmark with over 100,000 real patient heartbeats mapped to the 4 official AAMI medical categories:"
    p2.font.size = Pt(13)
    p2.font.color.rgb = C_BLUE_LIGHT

    cat_data = [
        ("Normal (N)", "90,113 beats (89.5%)", "Healthy sinus rhythm and typical heart patterns.", C_BLUE_LIGHT),
        ("Supraventricular (S)", "2,781 beats (2.8%)", "Abnormal beats starting early in the upper heart chambers (atria).", C_BLUE_ACCENT),
        ("Ventricular (V)", "7,008 beats (7.0%)", "Dangerous irregular contractions in the lower chambers (PVCs). Critical!", RGBColor(225, 29, 72)),
        ("Fusion (F)", "802 beats (0.8%)", "A rare, complex blend of normal and ventricular heartbeats.", RGBColor(217, 119, 6))
    ]

    for i, (name, count, desc, col) in enumerate(cat_data):
        cx = Inches(0.8 + i * 2.95)
        add_card(s4, cx, Inches(2.9), Inches(2.8), Inches(3.9))
        tb = s4.shapes.add_textbox(cx + Inches(0.15), Inches(3.05), Inches(2.5), Inches(3.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = name
        p1.font.size = Pt(15)
        p1.font.bold = True
        p1.font.color.rgb = col
        p1.space_after = Pt(8)

        p2 = tf.add_paragraph()
        p2.text = count
        p2.font.size = Pt(14)
        p2.font.bold = True
        p2.font.color.rgb = C_NAVY_DARK
        p2.space_after = Pt(12)

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(12)
        p3.font.color.rgb = C_TEXT_MUTED

    add_notes(s4,
        "We tested our model on the MIT-BIH Arrhythmia Database, which is the international benchmark for heart classification. "
        "It contains over 100,000 beats divided into 4 official categories: Normal beats, Supraventricular beats from the upper chambers, "
        "Ventricular beats from the lower chambers which can be fatal, and rare Fusion beats. As you can see, almost 90% of beats are normal, "
        "which makes finding the rare arrhythmias a true needle-in-a-haystack problem."
    )

    # =========================================================================
    # SLIDE 5: Preprocessing & Rate Encoding
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5, C_BG_LIGHT)
    add_header(s5, "Data Pipeline: From Raw Heartbeat to Brain Spikes")

    b_cards = [
        ("1. Slicing the Heartbeat", [
            "We locate each heartbeat by finding its main sharp spike (the R-peak).",
            "We crop a neat window of 128 points around it.",
            "This captures the complete heartbeat wave from start to finish.",
            "Ensures every beat fed into the AI has identical length."
        ], C_BLUE_LIGHT),
        ("2. Cleaning the Signal", [
            "Real heart signals get noisy when patients breathe, walk, or move.",
            "We apply a robust scaling filter that smoothly cleans up baseline drift.",
            "Scales every heartbeat neatly into a uniform 0 to 1 range.",
            "Ensures signals look clean and consistent for the neural network."
        ], C_BLUE_ACCENT),
        ("3. Converting to Spikes", [
            "How does an SNN read a heartbeat? Using 'Rate Coding' across 25 time steps.",
            "Think of it like Morse code:",
            "  • Tall peaks (QRS wave) → rapid, dense bursts of spikes.",
            "  • Flat baseline → silent, almost zero spikes.",
            "Translates the heart signal into a rhythmic language the AI understands."
        ], C_GREEN_ACC)
    ]

    for i, (ctitle, cbullets, col) in enumerate(b_cards):
        cx = Inches(0.8 + i * 3.95)
        add_card(s5, cx, Inches(1.6), Inches(3.8), Inches(5.2), border_color=col)
        tb = s5.shapes.add_textbox(cx + Inches(0.25), Inches(1.85), Inches(3.3), Inches(4.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = ctitle
        p_t.font.size = Pt(17)
        p_t.font.bold = True
        p_t.font.color.rgb = col
        p_t.space_after = Pt(16)

        for b in cbullets:
            pb = tf.add_paragraph()
            pb.text = "• " + b
            pb.font.size = Pt(13)
            pb.font.color.rgb = C_TEXT_MAIN
            pb.space_after = Pt(12)

    add_notes(s5,
        "Here is our data pipeline in three straightforward steps. First, we slice out each heartbeat by finding the sharp R-peak "
        "and taking 128 points around it. Second, because real-world signals have baseline wander from breathing and motion, "
        "we clean and normalize the signal between 0 and 1. Finally, we convert those values into spikes using rate coding over 25 timesteps: "
        "the tall peaks produce rapid bursts of spikes, while the flat resting baseline stays quiet. It is essentially translating heartbeats into Morse code."
    )

    # =========================================================================
    # SLIDE 6: Network Architecture (Simple Words, No Formulas)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6, C_BG_LIGHT)
    add_header(s6, "Network Architecture: Simple & Ultra-Compact")

    add_card(s6, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tb = s6.shapes.add_textbox(Inches(1.1), Inches(1.85), Inches(5.1), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Layer Structure: 128 → 32 → 32 → 4"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK
    p.space_after = Pt(14)
    arch_bullets = [
        "Input Layer (128 channels): One input for each point in the heartbeat window.",
        "Hidden Layer 1 (32 neurons): Detects basic wave shapes and early rhythm patterns.",
        "Hidden Layer 2 (32 neurons): Combines patterns to spot specific arrhythmia traits.",
        "Output Layer (4 neurons): One neuron for each category (Normal, Supraventricular, Ventricular, Fusion).",
        "Winner Takes All: Whichever output neuron fires the most spikes is our final prediction!"
    ]
    for b in arch_bullets:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(13)
        pb.font.color.rgb = C_TEXT_MAIN
        pb.space_after = Pt(10)

    add_card(s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), border_color=C_GREEN_ACC)
    tb2 = s6.shapes.add_textbox(Inches(7.1), Inches(1.85), Inches(5.1), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Why the Design is So Efficient"
    p2.font.size = Pt(17)
    p2.font.bold = True
    p2.font.color.rgb = C_GREEN_ACC
    p2.space_after = Pt(14)
    eff_bullets = [
        "Tiny Memory Footprint: Total network size is only 5,248 parameters (roughly 5.2 Kilobytes!).",
        "Runs on Simple Microchips: Can easily fit into tiny, ultra-cheap microcontrollers or wearable chips without external memory.",
        "No Heavy Math: Calculations use simple additions rather than thousands of floating-point multiplications.",
        "Fast Response: Analyzes each heartbeat in a fraction of a millisecond."
    ]
    for b in eff_bullets:
        pb = tf2.add_paragraph()
        pb.text = "✔ " + b
        pb.font.size = Pt(13)
        pb.font.color.rgb = C_TEXT_MAIN
        pb.space_after = Pt(10)

    add_notes(s6,
        "Our neural network layout is clean and minimal. It has 128 inputs, two hidden layers of 32 neurons each, and 4 output neurons. "
        "Whichever output neuron fires the most spikes represents the diagnosed class. The most impressive aspect is its size: "
        "the entire model has only 5,248 parameters—taking up barely 5 kilobytes. This means it can easily live on a tiny wearable chip "
        "without needing extra RAM or cloud processing."
    )

    # =========================================================================
    # SLIDE 7: Training: Overcoming the Spike Problem (No Math)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7, C_BG_LIGHT)
    add_header(s7, "Training the AI: Overcoming the Spike Problem")

    add_card(s7, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tb = s7.shapes.add_textbox(Inches(1.1), Inches(1.85), Inches(5.1), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "The Challenge: Spikes Don't Have Slopes"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGBColor(225, 29, 72)
    p.space_after = Pt(14)
    sg_bullets1 = [
        "Modern AI normally learns using calculus slopes (gradients).",
        "When an AI makes an error, it follows the slope backwards to tweak its weights.",
        "The Problem with Spikes: A spike is an instant jump from 0 to 1.",
        "Because it is an instant cliff, the slope is flat everywhere—which normally halts AI learning completely!",
        "For decades, this made training spiking networks with deep learning nearly impossible."
    ]
    for b in sg_bullets1:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(13)
        pb.font.color.rgb = C_TEXT_MAIN
        pb.space_after = Pt(10)

    add_card(s7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), border_color=C_GREEN_ACC)
    tb2 = s7.shapes.add_textbox(Inches(7.1), Inches(1.85), Inches(5.1), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "The Solution: Surrogate Gradients"
    p2.font.size = Pt(17)
    p2.font.bold = True
    p2.font.color.rgb = C_GREEN_ACC
    p2.space_after = Pt(14)
    sg_bullets2 = [
        "The Clever AI Trick: We separate forward prediction from backwards learning.",
        "When running forward: The neurons use real, discrete binary spikes.",
        "When learning backwards: The computer pretends the sharp cliff is a smooth, gentle hill.",
        "This allows standard optimizers (like Adam) to calculate corrections smoothly.",
        "Result: We get real biological spikes with the training power of modern deep learning!"
    ]
    for b in sg_bullets2:
        pb = tf2.add_paragraph()
        pb.text = "✔ " + b
        pb.font.size = Pt(13)
        pb.font.color.rgb = C_TEXT_MAIN
        pb.space_after = Pt(10)

    add_notes(s7,
        "Here is the core technical challenge we solved. Deep learning algorithms rely on calculus slopes to learn from their mistakes. "
        "However, a spike is an abrupt jump from 0 to 1 with no smooth slope, which used to prevent backpropagation in spiking networks. "
        "We used a technique called Surrogate Gradient Learning: during the forward pass, the model uses real discrete spikes. "
        "During backward training, it substitutes a smooth mathematical curve. This clever trick lets the AI learn effortlessly using modern backpropagation."
    )

    # =========================================================================
    # SLIDE 8: Teaching the Rhythm (Loss Function)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_bg(s8, C_BG_LIGHT)
    add_header(s8, "Teaching the Rhythm: Timing-Aware Learning")

    vr_cards = [
        ("Why Standard Loss Fails", [
            "Normal classification loss only checks: 'Did the model pick the right label at the end?'",
            "It completely ignores how quickly or steadily the pulses fired.",
            "Cardiac diagnosis needs timing awareness, not just a final vote."
        ]),
        ("Teaching Rhythm & Timing", [
            "We used a timing-aware loss function (Van Rossum distance).",
            "The model converts spike pulses into smooth activity waves over time.",
            "It compares its predicted rhythm against the correct rhythm at every single time step.",
            "Penalizes both missing a beat and firing too late."
        ]),
        ("The Training Result", [
            "Smooth and reliable learning across all 60 training epochs.",
            "Total error dropped smoothly by ~71% from start to finish.",
            "The network learned to fire quickly and decisively when an arrhythmia appears."
        ])
    ]

    for i, (ctitle, cbullets) in enumerate(vr_cards):
        cx = Inches(0.8 + i * 3.95)
        add_card(s8, cx, Inches(1.6), Inches(3.8), Inches(5.2))
        tb = s8.shapes.add_textbox(cx + Inches(0.25), Inches(1.85), Inches(3.3), Inches(4.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = ctitle
        p_t.font.size = Pt(17)
        p_t.font.bold = True
        p_t.font.color.rgb = C_BLUE_LIGHT
        p_t.space_after = Pt(16)

        for b in cbullets:
            pb = tf.add_paragraph()
            pb.text = "• " + b
            pb.font.size = Pt(13)
            pb.font.color.rgb = C_TEXT_MAIN
            pb.space_after = Pt(12)

    add_notes(s8,
        "To teach the network, we needed an objective function that cares about timing. Standard loss functions only ask whether the final answer "
        "was right, ignoring whether it fired early or late. We used a rhythm-aware metric called Van Rossum distance. "
        "It compares the continuous timing pattern of the model's spikes against the true rhythm across all 25 timesteps. "
        "This taught the network to fire cleanly and immediately when an arrhythmia occurs."
    )

    # =========================================================================
    # SLIDE 9: Results & Multi-Seed Benchmark
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_bg(s9, C_BG_LIGHT)
    add_header(s9, "Experimental Results: Multi-Seed Benchmark")

    rows = 3
    cols = 5
    tbl_shape = s9.shapes.add_table(rows, cols, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.5))
    table = tbl_shape.table

    table.columns[0].width = Inches(4.1)
    table.columns[1].width = Inches(1.9)
    table.columns[2].width = Inches(1.9)
    table.columns[3].width = Inches(1.9)
    table.columns[4].width = Inches(1.9)

    headers = ["Model Version", "Overall Accuracy", "Balanced Score (F1)", "Normal Detection", "Arrhythmia Precision"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY_DARK
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    data_rows = [
        ["Published SparrowSNN Baseline", "93.26 ± 0.82%", "71.14 ± 1.90%", "93.4%", "~50% (High false alarms)"],
        ["Our Optimized SNN AI Model 🚀", "97.24 ± 0.19%", "83.85 ± 1.36%", "98.57%", "~75% (Accurate & reliable)"]
    ]

    for i, row in enumerate(data_rows):
        is_highlight = (i == 1)
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            if is_highlight:
                cell.fill.fore_color.rgb = RGBColor(236, 253, 245)
            else:
                cell.fill.fore_color.rgb = RGBColor(255, 255, 255)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(13)
            if is_highlight:
                p.font.bold = True
                p.font.color.rgb = C_NAVY_DARK if j == 0 else C_GREEN_ACC
            else:
                p.font.color.rgb = C_TEXT_MAIN
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    stat_cards = [
        ("97.24% ± 0.19%", "Overall Accuracy", "+3.98% over standard baseline"),
        ("83.85% ± 1.36%", "Macro-F1 Score", "+12.71% balanced improvement"),
        ("± 0.19%", "Super Low Variance", "Consistently high across random seeds"),
        ("5.2 KB", "Total AI Size", "Fits on any microcontroller")
    ]

    for i, (val, title, sub) in enumerate(stat_cards):
        cx = Inches(0.8 + i * 2.95)
        add_card(s9, cx, Inches(4.5), Inches(2.8), Inches(2.2), border_color=C_GREEN_ACC)
        tb = s9.shapes.add_textbox(cx + Inches(0.15), Inches(4.65), Inches(2.5), Inches(1.9))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = val
        p1.font.size = Pt(20)
        p1.font.bold = True
        p1.font.color.rgb = C_GREEN_ACC
        p1.space_after = Pt(4)

        p2 = tf.add_paragraph()
        p2.text = title
        p2.font.size = Pt(12)
        p2.font.bold = True
        p2.font.color.rgb = C_NAVY_DARK
        p2.space_after = Pt(2)

        p3 = tf.add_paragraph()
        p3.text = sub
        p3.font.size = Pt(11)
        p3.font.color.rgb = C_TEXT_MUTED

    add_notes(s9,
        "Here are the final benchmark results across three independent test runs. The published baseline achieved 93.26% accuracy. "
        "Our optimized SNN surpassed this significantly, reaching 97.24% overall accuracy and an 83.85% balanced F1 score. "
        "Even more importantly, our standard deviation was just 0.19%, proving that the AI is rock solid and completely reproducible."
    )

    # =========================================================================
    # SLIDE 10: Clinical Significance
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_bg(s10, C_BG_LIGHT)
    add_header(s10, "Clinical Impact: Slashing False Alarms")

    add_card(s10, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tb = s10.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Test Confusion Matrix (Over 20,000 Beats)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK
    p.space_after = Pt(14)

    cm_text = (
        "               Pred N     Pred S     Pred V     Pred F\n"
        "  True N:     17,805         79        118         21\n"
        "  True S:        165        363         28          0\n"
        "  True V:         49          4      1,339         10\n"
        "  True F:         24          0         14        122\n\n"
        "Key Highlights:\n"
        "• Healthy Beat Accuracy: 98.79% (Rarely causes false alarms)\n"
        "• Dangerous PVC Detection: 95.51% (Catches life-threatening beats)\n"
        "• High Arrhythmia Precision: ~75% to 81% true alert rate"
    )
    p2 = tf.add_paragraph()
    p2.text = cm_text
    p2.font.name = "Consolas"
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_BLUE_ACCENT

    add_card(s10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tb2 = s10.shapes.add_textbox(Inches(7.1), Inches(1.85), Inches(5.1), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Why This Matters in Healthcare"
    p2.font.size = Pt(17)
    p2.font.bold = True
    p2.font.color.rgb = C_BLUE_LIGHT
    p2.space_after = Pt(14)
    clin_bullets = [
        "Cutting False Alarms by ~46%: Drops nuisance alarms on healthy beats by nearly half, directly solving 'alarm fatigue' for nurses and doctors.",
        "Clinical Trust: When this system sounds an alarm, doctors know there is a 75-80% probability it is genuine, compared to only ~50% in prior models.",
        "Critical Sensitivity: Catches premature ventricular contractions with over 95.5% reliability, preventing progression to fatal ventricular fibrillation.",
        "Edge-Ready: All of this happens instantly on the patient's device."
    ]
    for b in clin_bullets:
        pb = tf2.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(13)
        pb.font.color.rgb = C_TEXT_MAIN
        pb.space_after = Pt(12)

    add_notes(s10,
        "Looking at the test confusion matrix on over 20,000 unseen patient beats: Normal beat accuracy is nearly 99%. "
        "In healthcare AI, false alarms are a major problem—if a monitor beeps constantly, patients ignore it. "
        "Our model reduced false alarms by nearly 46% while catching over 95.5% of dangerous ventricular arrhythmias. "
        "This proves our neuromorphic SNN delivers clinical-grade accuracy with ultra-low power and a tiny 5.2 KB footprint. "
        "Thank you, and I am happy to answer any questions."
    )

    output_path = "Neuromorphic_AI_ECG_Simplified.pptx"
    try:
        prs.save("Neuromorphic_AI_ECG_Classification.pptx")
        output_path = "Neuromorphic_AI_ECG_Classification.pptx"
    except PermissionError:
        prs.save(output_path)

    print(f"[OK] Successfully generated presentation: {os.path.abspath(output_path)}")
    return output_path

if __name__ == "__main__":
    create_presentation()
