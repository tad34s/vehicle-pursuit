using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class TrackGenerator : MonoBehaviour
{
	public GameObject marker;
	public GameObject parentCheckpoints;
	public List<GameObject> checkpoints;

	public SplineCreator sc;

	public GameObject agent;
	private AgentCar carAgent;

	private GameObject currentRoad = null;

	public int triesPerTrack = 2;
	private int tryCount = 1;
	private bool firstInit = true;

	public void Start()
	{
		/*
		 * Beginning
		 *  Spline cell count: 2
		 *  Num of sites to generate: 10
		 *  Spline Scale: 0.8
		 *  Size of voronoi: 200
		 * 
		 * End:
		 *  Spline cell count: 3
		 *  Num Of Sites to generate: 20
		 *  Spline scale: 1
		 *  Size of voronoi: 200
		 */
		sc.Init();

		carAgent = agent.GetComponent<AgentCar>();
	}

    public void Init()
    {
		// Debug.Log("Init track");

		Mesh roadMesh;

		if(tryCount >= triesPerTrack || firstInit)
		{
            RemoveTrack();

            currentRoad = GenerateTrack();
            roadMesh = currentRoad.GetComponent<MeshFilter>().mesh;

            // RemoveMarkers();
            CreateMarkers(roadMesh);

			tryCount = 1;
			firstInit = false;
		}
		else
		{
			roadMesh = currentRoad.GetComponent<MeshFilter>().mesh;
			tryCount++;
		}
		PlaceAgent(roadMesh);
    }

	private void RemoveTrack()
	{
		// Debug.Log("Removing track");
		if (currentRoad != null)
		{
			Destroy(currentRoad);
			currentRoad = null;
		}

		// Debug.Log("Removed track");
	}
	
	private void RemoveMarkers()
	{
		// Debug.Log("Removing markers");
		foreach(GameObject child in checkpoints)
			Destroy(child);
		// Debug.Log("Removed markers");
	}

    private void PlaceAgent(Mesh roadMesh)
	{
		// Debug.Log("Placing agent");

		Vector3[] vertices = roadMesh.vertices;
		float yLevel = 0.01f;

		Transform firstCheckpoint = checkpoints[0].transform;
		Transform secondCheckpoint = checkpoints[1].transform;

		Vector3 startingPosition = new Vector3(
			firstCheckpoint.position.x,
			yLevel,
			firstCheckpoint.position.z
		);

		Vector3 nextPos = new Vector3(
			secondCheckpoint.position.x,
			yLevel,
			secondCheckpoint.position.z
		);

		Vector3 targetDir = (nextPos - startingPosition);

		float angle = Vector3.SignedAngle(targetDir, Vector3.forward, Vector3.up);
		// Debug.Log("Angle: " + angle);
		Quaternion rotation = Quaternion.Euler(0, -angle, 0);

		agent.transform.position = startingPosition;
		agent.transform.rotation = rotation;

		// Debug.Log("Placed agent");
	}

	private GameObject GenerateTrack()
	{
		// Debug.Log("Generating track");
		sc.GenerateVoronoi();
		sc.GenerateSpline();
		GameObject track = sc.CreateTrack();

		// Debug.Log("Generated track");
		return track;
	}

	private void CreateMarkers(Mesh roadMesh)
	{
		// Debug.Log("Creating markers");
		Vector3[] vertices = roadMesh.vertices;
		int checkpointIndex = 0;
		for (int i = 0; i < vertices.Length - 3; i += 2, checkpointIndex++)
		{
			Vector3 markerPosition = new Vector3(
				(vertices[i].x + vertices[i + 3].x) / 2,
				parentCheckpoints.transform.position.y,
				(vertices[i].z + vertices[i + 3].z) / 2
			);

			if(checkpointIndex < checkpoints.Count)
			{
				checkpoints[checkpointIndex].transform.position = markerPosition;
			} else
			{
                GameObject ob = Instantiate(marker, markerPosition, Quaternion.identity, parentCheckpoints.transform);
                checkpoints.Add(ob);
			}
		}

		if(checkpointIndex < checkpoints.Count)
		{
			for(int i = checkpointIndex; i < checkpoints.Count; i++)
			{
				Destroy(checkpoints[i]);
			}
			checkpoints.RemoveRange(checkpointIndex, checkpoints.Count - checkpointIndex);
		}

		// Debug.Log("Created markers");
	}
}
