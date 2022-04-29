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
    public EyeLogger eyeLogger;
    public LayerMask lookableObjectLayer;
    public string lookableObjectTag;

    private Vector3 _gazeVector; //normalized
    private Vector3 _gazePoint;
    private string _gazeObjectName;

    public bool shouldCalibrate;

    void Start()
    {
        if (!SRanipal_Eye_Framework.Instance.EnableEye)
            enabled = false;

        if(shouldCalibrate)
            SRanipal_Eye.LaunchEyeCalibration();
    }

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
        
        //If we are getting eye data from the callback instead of Update, halt Update()
        if(SRanipal_Eye_Framework.Instance.EnableEyeDataCallback)
            return;
        
        //Initialize values
        //TODO: what do we do with these values where no collider is hit by the ray?
        _gazePoint = Vector3.positiveInfinity;
        _gazeObjectName = "";

        //Gets the Gaze Ray
        SRanipal_Eye.GetVerboseData(out var data);
        _gazeVector = data.combined.eye_data.gaze_direction_normalized;
        
        //Casts the Gaze Ray casted and returns the position of the object hit
        Ray gazeRay;
        bool eye_focus;
        if (eye_callback_registered)
            eye_focus = SRanipal_Eye.Focus(GazeIndex.COMBINE, out gazeRay, out FocusInfo, 0, MaxDistance, lookableObjectLayer, eyeData);
        else
            eye_focus = SRanipal_Eye.Focus(GazeIndex.COMBINE, out gazeRay, out FocusInfo, 0, MaxDistance, lookableObjectLayer);
        
        if (eye_focus)
        {
            if (FocusInfo.transform.gameObject.CompareTag("LookableObject") || FocusInfo.transform.gameObject.CompareTag("Shelf"))
            {
                _gazePoint = FocusInfo.transform.position;
                _gazeObjectName = FocusInfo.transform.name;
            }
        }
        else
        {
            _gazePoint = gazeRay.GetPoint(MaxDistance);
            _gazeObjectName = "";
        }
        
        eyeLogger.Log(_gazeVector, _gazePoint, _gazeObjectName);
        
    }
    
    private void EyeCallback(ref EyeData eye_data)
    {
        /*if(!SRanipal_Eye_Framework.Instance.EnableEyeDataCallback || !_started)
            return;
        
        //Initialize values
        _gazePoint = Vector3.positiveInfinity;
        _gazeObjectName = "";

        //Gets the Gaze Ray
        _gazeVector = eye_data.verbose_data.combined.eye_data.gaze_direction_normalized;
        
        //Casts the Gaze Ray casted and returns the position of the object hit
        Ray gazeRay;
        bool eye_focus;
        if (eye_callback_registered)
            eye_focus = SRanipal_Eye.Focus(GazeIndex.COMBINE, out gazeRay, out FocusInfo, 0, MaxDistance, lookableObjectLayer, eyeData);
        else
            eye_focus = SRanipal_Eye.Focus(GazeIndex.COMBINE, out gazeRay, out FocusInfo, 0, MaxDistance, lookableObjectLayer);
        
        if (eye_focus)
        {
            if (FocusInfo.transform.gameObject.CompareTag(lookableObjectTag))
            {
                _gazePoint = FocusInfo.transform.position;
                _gazeObjectName = FocusInfo.transform.name;
            }
        }
        
        if(_gazeObjectName != "")
            Debug.Log($"From EyeManager CALLBACK, logging {_gazeVector}, {_gazePoint}, {_gazeObjectName}");
        */
    }
    
}
