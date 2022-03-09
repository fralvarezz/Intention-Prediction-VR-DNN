using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.XR;

public class TrackerData : MonoBehaviour
{
    private InputDevice tracker;
    
    // Start is called before the first frame update
    void Start()
    {
        var allDevices = new List<InputDevice>();
        InputDevices.GetDevices(allDevices);
        tracker = allDevices.FirstOrDefault(d => d.role == InputDeviceRole.HardwareTracker);
    }

    // Update is called once per frame
    void Update()
    {
        tracker.TryGetFeatureValue(CommonUsages.devicePosition, out var pos);
        tracker.TryGetFeatureValue(CommonUsages.deviceRotation, out var rot);
        
        transform.position = pos;
        transform.rotation = rot;
    }
}
