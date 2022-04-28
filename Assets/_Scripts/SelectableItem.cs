using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SelectableItem : MonoBehaviour
{
    public bool isSelected = false;
    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player") && isSelected)
        {
            ExperimentManager.instance.UnHighlightItem();
        }
    }
}
