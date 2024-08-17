using UnityEngine;

namespace Unity.MLAgents.Areas
{
	[DefaultExecutionOrder(-5)]
	public class TrainingReplicator : MonoBehaviour
	{

		public GameObject baseArea;
		public int numAreas = 1;
		public float margin = 20;

		public enum RoadColor{
			Amazon,
			BlackWhite
		};

		// Wide: 15
		// Slim: 10
		public int roadSize = 15;
		public RoadColor roadColor;
		public bool cameraGrayscale = false;
		public int cameraWidth = 64;
		public int cameraHeight = 64;

		public void Awake()
		{
			if (Academy.Instance.IsCommunicatorOn){
				numAreas = Academy.Instance.NumAreas;

				// TODO: Set env variables
				// TODO: Set camera and texture size
			}
		}

		void OnEnable()
		{
			AddAreas();
		}

		private void AddAreas()
		{
			for(int i = 0; i < numAreas; i++)
			{
				if (i == 0)
					continue;

				Vector3 pos = Vector3.up * margin * i;
				Instantiate(baseArea, pos, Quaternion.identity);
			}
		}
	}

}
