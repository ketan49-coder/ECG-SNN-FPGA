# CardioSpike-FPGA: Master Defense Cheat Sheet
*An end-to-end plain English guide of the entire project, from raw data to final hardware diagnosis.*

---

## PHASE 1: THE DATA (Python)
*How we get the heartbeats ready for the network.*

1. **The Window:** MIT-BIH records ECGs at 360 samples per second. We isolate a single heartbeat by grabbing the R-Peak and taking 64 samples before it and 63 samples after it. 
   * *Result:* A 128-sample window (representing 0.35 seconds of time).
2. **Normalization:** Heartbeats have different voltages depending on the patient. We use Min-Max Normalization to stretch/squash every heartbeat so its lowest point is `0` and its highest peak is `1.0` (which becomes `0 to 255` in 8-bit hardware).
3. **Balancing:** 80% of the dataset is "Normal." If we train on this, the network just guesses "Normal" every time. We use `augment_data.py` to add noise and shift the rare diseases so the network is forced to learn them.

---

## PHASE 2: THE TRAINING (Python)
*How the network learns the weights.*

1. **The Architecture:** 128 inputs &rarr; 16 hidden neurons &rarr; 16 hidden neurons &rarr; 5 output neurons (the 5 AAMI classes: N, S, V, F, Q).
2. **BPTT & Surrogate Gradients:** Spikes are binary (`0` or `1`), which breaks standard calculus. We use *Surrogate Gradients* (pretending the spike is a smooth curve) so we can use *Backpropagation Through Time (BPTT)* to trace errors backward over 16 timesteps and tweak the weights.
3. **Quantization:** Python calculates weights using heavy 32-bit floating-point decimals. We quantize (compress) them into 8-bit fixed-point numbers (Q1.7 format). This allows us to fit all 2,384 weights perfectly into a single Block RAM (BRAM) on the FPGA.

---

## PHASE 3: THE FPGA HARDWARE (VHDL)
*The physical chip running the inference. This loops 16 times (Timesteps) for a single heartbeat.*

**Step 1: The Input Registers**
The 128 normalized samples (values 0-255) are loaded onto the 128 input pins of the FPGA. They sit there, frozen.

**Step 2: The Rate Encoder (LFSR)**
* *The Goal:* Turn 8-bit numbers into 1-bit spikes so we don't need multipliers.
* *How:* An LFSR generates a random 8-bit number (the "bar"). If the patient's sample is greater than the random bar, it fires a `'1'`. If it's less, it fires a `'0'`. 

**Step 3: The Synapses (Weights)**
The `'1'`s and `'0'`s flow to the first layer of 16 neurons. If a wire carries a `'1'`, the hardware adds that wire's specific *Weight* into the neuron. If it carries a `'0'`, it does nothing.

**Step 4: The LIF Neuron (The Bucket)**
Each neuron acts like a bucket accumulating voltage (`v_mem`). 
* *The Leak:* Every timestep, the bucket leaks a little bit of voltage. Instead of using division (which is expensive in hardware), we do an **arithmetic right-shift (`v_mem >> 4`)**, which easily subtracts ~6% of the voltage.
* *The Fire:* If the bucket fills up past the Threshold, the neuron fires a `'1'` to the next layer, and the bucket resets to `0` (Zero Reset).

**Step 5: The FSM (State Machine)**
If we calculated all neurons at the exact same nanosecond, the FPGA would run out of physical logic gates. The FSM acts like a traffic cop, calculating Neuron 0, then Neuron 1, then Neuron 2, sequentially. This saves massive amounts of physical space.

**Step 6: The Argmax (The Final Decision)**
After looping through Steps 2-5 for 16 timesteps, the 5 Output Neurons have been firing spikes into 5 digital counters. The Argmax is a simple combinational logic circuit that compares the 5 counters, finds the maximum number, and outputs the final disease classification (0, 1, 2, 3, or 4).

---

## CORE DEFENSE QUESTIONS & ANSWERS

**Q: Why Spiking Neural Networks (SNNs) instead of CNNs?**
*A:* CNNs require power-hungry DSP (Digital Signal Processing) multipliers. Because SNNs use binary spikes, we replace all multiplication with simple addition, saving massive amounts of battery power for wearable edge devices.

**Q: Why Rate Coding instead of Temporal Coding?**
*A:* Temporal coding requires complex digital timers on every single input pin to track exactly *when* a spike fires. Rate coding allows us to use one shared random number generator (LFSR), which takes up vastly less physical space on the FPGA.

**Q: How did you implement the LIF Leak without multipliers?**
*A:* We mathematically restricted the leak constant to `0.9375` (which is `15/16`). In binary hardware, subtracting `1/16th` is achieved by simply bit-shifting the binary number to the right by 4 spaces (`>> 4`). This completely eliminates the need for hardware division or multiplication.


---

## PHASE 4: FUTURE WORK (The Grant Proposal)
*The pitch for the 1-Crore MeitY/DRDO Grant (Moving from BPTT to SuperSpike).*

**The Limitation of Phase 1:** 
Currently, our weights are frozen. Because we used BPTT (Backpropagation Through Time), the network had to be trained offline on a massive Python GPU. If a patient's heart changes over time, or if we deploy this to an offshore oil rig, the chip cannot adapt to new environments.

**The Phase 2 Moonshot (SuperSpike):** 
For our grant proposal, we will upgrade the VHDL architecture to support **SuperSpike (Zenke's 3-Factor Rule)**. SuperSpike allows for supervised, "forward-in-time" learning directly on the physical chip. 

By calculating three local factors in real-time on the FPGA:
1. **The Error (`e`)**: Did the network make a mistake?
2. **The Surrogate Gradient (`σ`)**: Is the neuron at the critical threshold to learn?
3. **The Eligibility Trace**: Did this specific input wire spike recently?

We can build an architecture that learns autonomously at the edge without *ever* needing an external computer or cloud connection. This is the holy grail of neuromorphic engineering.
