using UnityEngine;

namespace Unity.MLAgents.Areas
{
    [DefaultExecutionOrder(-5)]
    public class TrainingReplicator : MonoBehaviour
    {
        public GameObject baseArea;
        public GameObject leaderArea;

        private bool leaderOnly;

        public int numAreas = 1;
        public float margin = 20;

        private GameObject[] carCameras;

        public enum RoadColor
        {
            Amazon,
            BlackWhite,
        };

        // Wide: 15
        // Slim: 10
        public int roadSize = 15;
        public RoadColor roadColor;

        public Color backgroundColor;

        public bool randomBackgroundColor;
        private float changingColorSpeed;
        private bool displayRandomColorInMain = false;

        public Material[] roadMaterials;

        struct HSV
        {
            public float hue;
            public float saturation;
            public float value;
        };

        private HSV hsvColor;


        public void Awake()
        {
            if (Academy.Instance.IsCommunicatorOn)
            {
                numAreas = Academy.Instance.NumAreas;
            }
        }

        void Start()
        {
            roadSize = DataChannel.getParameter("roadSize", 15);
            roadColor = (RoadColor)DataChannel.getParameter("roadColor", 0);
            randomBackgroundColor = System.Convert.ToBoolean(
                DataChannel.getParameter("randomBackgroundColor", 0)
            );
            leaderOnly = DataChannel.getParameter("leaderOnly", 0) >= 1;

            if (!randomBackgroundColor)
                backgroundColor = DataChannel.getParemeter(
                    "backgroundColor",
                    new Color(0, 0.819607843f, 0.529411765f)
                );
            else
            {
                hsvColor.hue = Random.Range(0f, 1f);
                hsvColor.saturation = 0.25f;
                hsvColor.value = 1f;
                backgroundColor = Color.HSVToRGB(hsvColor.hue, .5f, hsvColor.value);

                changingColorSpeed = DataChannel.getParameter(
                    "changingBackgroundColorSpeed",
                    0.75f
                );
            }

            SetMaterial();

            SetAreas();

            carCameras = GameObject.FindGameObjectsWithTag("CarCamera");

            ChangeCameraBackgroundColor();
        }

        void Update()
        {
            if (randomBackgroundColor)
            {
                if (Input.GetKeyDown(KeyCode.Escape))
                    displayRandomColorInMain = !displayRandomColorInMain;

                backgroundColor = Color.HSVToRGB(
                    (hsvColor.hue + Time.time * changingColorSpeed) % 1f,
                    hsvColor.saturation + (Mathf.Sin(Time.time) * 0.25f + 0.25f),
                    hsvColor.value
                );
                ChangeCameraBackgroundColor();
            }
        }

        private void ChangeCameraBackgroundColor()
        {
            foreach (GameObject c in carCameras)
            {
                c.GetComponent<Camera>().backgroundColor = backgroundColor;
            }

            if (displayRandomColorInMain || !randomBackgroundColor)
                GameObject.Find("Camera").GetComponent<Camera>().backgroundColor = backgroundColor;
        }

        void SetMaterial()
        {
            foreach (var mat in roadMaterials)
            {
                mat.SetFloat(
                    "_ReflectionStrength",
                    DataChannel.getParameter("reflectionStrength", 0.4f)
                );
                mat.SetFloat("_NoiseScaleX", DataChannel.getParameter("noiseScaleX", 0.04f));
                mat.SetFloat("_NoiseScaleY", DataChannel.getParameter("noiseScaleY", 0.05f));
                mat.SetFloat("_Speed", DataChannel.getParameter("noiseSpeed", 0.2f));
            }
        }

        private void SetAreas()
        {
            Debug.Log($"Leader only: {leaderOnly}");
            
            if(leaderOnly){
                EnableAreaComponents(leaderArea, true);
                EnableAreaComponents(baseArea, false);
            } else {
                EnableAreaComponents(leaderArea, false);
                EnableAreaComponents(baseArea, true);
            }

            // Only instantiate additional areas from the selected area
            GameObject selectedArea = leaderOnly ? leaderArea : baseArea;
            for (int i = 0; i < numAreas; i++)
            {
                if (i == 0)
                    continue;

                Vector3 pos = Vector3.up * margin * i;
                Instantiate(
                        selectedArea,
                        pos,
                        Quaternion.identity
                    );
            }
        }

        private void EnableAreaComponents(GameObject area, bool enable)
        {
            // Disable/Enable all Agent components in the area
            Agent[] agents = area.GetComponentsInChildren<Agent>();
            foreach (Agent agent in agents)
            {
                agent.enabled = enable;
            }

            // Disable/Enable all cameras in the area to prevent rendering
            Camera[] cameras = area.GetComponentsInChildren<Camera>();
            foreach (Camera camera in cameras)
            {
                camera.enabled = enable;
            }

            area.SetActive(enable);
        }
    }
}
