using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using ViveSR.anipal.Eye;

public class EyeManager : MonoBehaviour
{
    
    private readonly GazeIndex[] GazePriority = new GazeIndex[] { GazeIndex.COMBINE, GazeIndex.LEFT, GazeIndex.RIGHT };
    private static EyeData eyeData = new EyeData();
    private bool eye_callback_registered = false;
    private FocusInfo FocusInfo;
    private readonly float MaxDistance = 20;

    // Start is called before the first frame update
    void Start()
    {
        if (!SRanipal_Eye_Framework.Instance.EnableEye)
        {
            enabled = false;
        }   
    }

    // Update is called once per frame
    void Update()
    {
        if (SRanipal_Eye_Framework.Status != SRanipal_Eye_Framework.FrameworkStatus.WORKING &&
            SRanipal_Eye_Framework.Status != SRanipal_Eye_Framework.FrameworkStatus.NOT_SUPPORT) return;

        if (SRanipal_Eye_Framework.Instance.EnableEyeDataCallback && !eye_callback_registered)
        {
            SRanipal_Eye.WrapperRegisterEyeDataCallback(Marshal.GetFunctionPointerForDelegate((SRanipal_Eye.CallbackBasic)EyeCallback));
            eye_callback_registered = true;
        }
        else if (!SRanipal_Eye_Framework.Instance.EnableEyeDataCallback && eye_callback_registered)
        {
            SRanipal_Eye.WrapperUnRegisterEyeDataCallback(Marshal.GetFunctionPointerForDelegate((SRanipal_Eye.CallbackBasic)EyeCallback));
            eye_callback_registered = false;
        }
        
        Ray GazeRay;
        int lookableObjectLayer = LayerMask.NameToLayer("LookableObject");
        bool eye_focus;
        if (eye_callback_registered)
            eye_focus = SRanipal_Eye.Focus(GazeIndex.COMBINE, out GazeRay, out FocusInfo, 0, MaxDistance, (1 << lookableObjectLayer), eyeData);
        else
            eye_focus = SRanipal_Eye.Focus(GazeIndex.COMBINE, out GazeRay, out FocusInfo, 0, MaxDistance, (1 << lookableObjectLayer));

        if (eye_focus)
        {
            Debug.Log(FocusInfo.point);
            
            if (FocusInfo.transform.gameObject.CompareTag("LookableObject"))
            {
                FocusInfo.transform.GetComponent<CubeRayChecker>().OnHit();
            }
            //FocusInfo.transform.GetComponent<CubeRayChecker>();
            //DartBoard dartBoard = FocusInfo.transform.GetComponent<DartBoard>();
            //if (dartBoard != null) dartBoard.Focus(FocusInfo.point);
        }
    }
    
    private static void EyeCallback(ref EyeData eye_data)
    {
        eyeData = eye_data;
    }
    
}
