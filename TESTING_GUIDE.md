# ARDUINO UNO Q4GB AI ROBOT - TESTING SCRIPTS

## QUICK TESTS (Run in order)

### 1. Camera and YOLO Test (Core Functionality)
```bash
python arduino_uno_q4gb_ai_robot\python_tools\testing\simple_camera_test.py --test
```
**Expected**: Camera opens with DirectShow/MSMF fallback, YOLO model loads, object detection works
**Success**: ✅ Camera initialized, ✅ YOLO model loaded, ✅ Real-time detection display

---

### 2. Hybrid AI Controller Test (Rule + Qwen)
```bash
python arduino_uno_q4gb_ai_robot\python_tools\testing\camera_only_tester.py --test
```
**Expected**: Hybrid AI controller with rule-based navigation + Qwen precision for trash objects
**Success**: ✅ Controller loads, ✅ Mode switching works, ✅ Servo commands generated

---

### 3. Complete System Validation
```bash
python final_test.py
```
**Expected**: All components validated together
**Success**: ✅ 3/3 tests pass, ✅ System ready for Arduino deployment

## PRECISION TESTING SCENARIOS

### Setup for Testing
1. **Place objects on table** at camera distance
2. **Test objects**: 
   - **Navigation objects**: Person, toy car, book (should use rule-based)
   - **Trash objects**: Bottle, cup, can, plastic (should trigger Qwen precision)
3. **Lighting**: Moderate indoor light works best

### Test Sequence
1. **Clear path test** - No objects → Should show "FORWARD" with rule-based mode
2. **Navigation test** - Show person/toy → Should follow with rule-based mode  
3. **Precision test** - Show bottle/cup → Should activate Qwen precision mode
4. **Mixed objects** - Person + trash nearby → Should prioritize precision for trash

## INTERPRETING RESULTS

### SUCCESS INDICATORS
✅ **Camera working**: DirectShow/MSMF backend message + resolution
✅ **YOLO functional**: "Model loaded" + class count + object boxes on screen
✅ **AI decisions**: Action text + confidence + command string
✅ **Mode switching**: "rule_based" for navigation, "qwen_precision" for trash

### Arduino Commands Generated
The system outputs servo commands in Arduino format:
```
CMD L160 R160 T500    # Forward
CMD L120 R180 T300    # Turn right  
CMD L0 R0 T200      # Stop
CMD L140 R100 T400    # Precision approach
```

## TROUBLESHOOTING

### Camera Issues
❌ If camera fails:
- Check if other apps use camera (Zoom, Teams, etc.)
- Try different camera IDs: --camera 1, --camera 2
- Check USB connections for external cameras

### YOLO Issues  
❌ If "YOLO not found":
- Ensure models/yolo26n.pt exists (5.5MB)
- Try from arduino_uno_q4gb_ai_robot directory
- Check Ultralytics installation: py -c "import ultralytics; print('✅')"

### Qwen Issues
❌ If "Qwen not available":
- Install transformers: py -m pip install transformers
- Check models/qwen_prompt.txt exists
- System will fall back to rule-based (still functional)

### Performance Issues
❌ If slow or laggy:
- Close other applications
- Use lower camera resolution in code
- Check CPU usage during testing

## NEXT STEPS AFTER TESTING

### 1. Arduino Hardware Connection
```bash
# Check available ports
python -c "import serial.tools; print([port.device for port in serial.tools.comports()])"

# Test Arduino connection
python -m serial.tools.miniterm --port COM3 --baud 115200
```

### 2. Firmware Upload
```bash
# Open Arduino IDE 2.0+
# Load: arduino_firmware/core/ai_robot_controller.ino
# Select Board: "Arduino UNO Q4GB"
# Upload to Arduino
```

### 3. Full Integration Test
```bash
# Run with real Arduino connected
python arduino_uno_q4gb_ai_robot\python_tools\testing\camera_only_tester.py --test --port COM3
```

This testing sequence ensures your Arduino UNO Q4GB AI robot is fully functional before hardware deployment.