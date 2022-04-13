using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Valve.VR;

public class InputManager : MonoBehaviour
{
    public SteamVR_ActionSet actionSet;

    public SteamVR_Action_Boolean booleanAction;

    public HandController handController;
    
    void Start()
    {
        actionSet.Activate(SteamVR_Input_Sources.Any, 0, true);
    }

    void Update()
    {
        if (booleanAction.stateDown)
        {
            string interactedObjectName = "";
            if (handController.collidingWith != null)
                interactedObjectName = handController.collidingWith.name;

            EyeLogger.Instance.SetInteractedObject(interactedObjectName);
            //do something
        }
    }
}
