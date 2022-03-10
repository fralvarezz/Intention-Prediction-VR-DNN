using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace ViveHandTracking.Sample {
  class DetectionStatus : MonoBehaviour {
    public TextMesh text3d = null;
    public Text text2d = null;

    void Update() {
      if (GestureProvider.Current != null && text3d != null) {
        transform.position = GestureProvider.Current.transform.position;
        transform.rotation = GestureProvider.Current.transform.rotation;
      }
      string text = "";
      if (GestureProvider.Status != GestureStatus.Running || text2d != null)
        text = "Hand Tracking Status: " + GestureProvider.Status.ToString();
      if (text2d != null)
        text2d.text = text;
      else if (text3d != null)
        text3d.text = text;
    }
  }

}
