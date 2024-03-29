import { Ai } from '@cloudflare/ai';
import { QdrantClient } from '@qdrant/js-client-rest';
export interface Env {
	VECTORIZE_INDEX: VectorizeIndex;
	AI: any;
}
interface EmbeddingResponse {
	shape: number[];
	data: number[][];
}

export default {
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
		const ai = new Ai(env.AI);
		let path = new URL(request.url).pathname;
		if (path.startsWith('/favicon')) {
			return new Response('', { status: 404 });
		}
		let h = Object.fromEntries(request.headers);

		if (env.API_KEY && h['api-key'] === env.API_KEY.trim()) {

			if (path === '/health') {
				return new Response('', { status: 200 });
			}

			if (path === '/insertraw')
			{
				let json = await request.json();
				let    inserted = await env.VECTORIZE_INDEX.upsert(json);
				return Response.json(inserted);
			}

			// You only need to generate vector embeddings once (or as
			// data changes), not on every request
			if (path === '/insert') {
				// In a real-world application, you could read content from R2 or
				// a SQL database (like D1) and pass it to Workers AI
				const stories = ['This is a story about an orange cloud', 'This is a story about a llama', 'This is a story about a hugging emoji'];
				const modelResp: EmbeddingResponse = await ai.run('@cf/baai/bge-base-en-v1.5', {
					text: stories,
				});

				// Convert the vector embeddings into a format Vectorize can accept.
				// Each vector needs an ID, a value (the vector) and optional metadata.
				// In a real application, your ID would be bound to the ID of the source
				// document.
				let vectors: VectorizeVector[] = [];
				let id = 1;
				modelResp.data.forEach((vector) => {
					vectors.push({ id: `${id}`, values: vector });
					id++;
				});

				let inserted = await env.VECTORIZE_INDEX.upsert(vectors);
				return Response.json(inserted);
			}
		}
       
		// Your query: expect this to match vector ID. 1 in this example
		let userQuery =  new URL(request.url).searchParams.get('q') || '';
		if (!userQuery) {
			return new Response('Please provide a query parameter q', { status: 400 });
		}

        let start = performance.now();
		const queryVector: EmbeddingResponse = await ai.run('@cf/baai/bge-base-en-v1.5', {
			text: [userQuery],
		});
        let mid = performance.now();
		//let matches2 = await env.VECTORIZE_INDEX.query(queryVector.data[0], { topK: 5 });
		const client = new QdrantClient({
			url: env.QDRANT_ENDPOINT, 
			apiKey: env.QDRANT_API_KEY 
		  });
		let searchResult = await client.search("posts", {
				vector: queryVector.data[0],
				limit: 10,	
		});

		let matches = {
			matches: searchResult.map((item) => ({
				id: item.payload!.id,
				score: item.score,			
			}))
		};

        let end = performance.now();
		return Response.json({
			// Expect a vector ID. 1 to be your top match with a score of
			// ~0.896888444
			// This tutorial uses a cosine distance metric, where the closer to one,
			// the more similar.
            embedDuration: mid - start,
            queryDuration: end - mid,
            q: userQuery,
			matches: matches,			
		});
	},
};