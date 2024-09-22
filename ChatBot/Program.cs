using Cloud.Unum.USearch; //vector storage database
using Microsoft.Data.Sqlite; //to read the provided datasource of Q&A's
using Dapper; //Helpers to work with System.Data
using OpenAI.Chat; 
using OpenAI.Embeddings;
using System.Text;


namespace ChatBot;


internal class Program
{
    static async Task Main(string[] args)
    {
        string database = @"dataset.db";
        string vectorDatabase = @"dataset.vdb";
        //Sqlite connection
        await using var db = new SqliteConnection($"Data Source={database}");
        //it was converted from parquet file, name is called converted
        //original dataset retrieved from: https://huggingface.co/datasets/HuggMaxi/preguntasEtiquetadas
        var questions = db.Query<Questions>("SELECT * FROM converted");


        //client used for generating responses
        ChatClient client = new(model: "gpt-4o-mini", Environment.GetEnvironmentVariable("OPENAI_API_KEY"));
        //client used for generting embeddings
        EmbeddingClient embeddingClient = new EmbeddingClient("text-embedding-3-small", Environment.GetEnvironmentVariable("OPENAI_API_KEY"));



        USearchIndex index;
        if (!File.Exists(vectorDatabase))
        {
            //if the embeddings database doesnt exist, we create one compatible with text-embedding-3-small (1536 size)
            index = new USearchIndex(
                metricKind: MetricKind.Cos, // Choose cosine metric
                quantization: ScalarKind.Float32, // Only quantization to Float32, Float64 is currently supported
                dimensions: 1536,  // Define the number of dimensions in input vectors
                connectivity: 16, // How frequent should the connections in the graph be, optional
                expansionAdd: 128, // Control the recall of indexing, optional
                expansionSearch: 64 // Control the quality of search, optional
            );
            Console.WriteLine($"Creating Vector Database");
            foreach (var question in questions)
            {
                string qAndA = $"{question.Consulta}\n{question.Texto}";
                //we grab the embeds
                var response = await embeddingClient.GenerateEmbeddingAsync(qAndA);
                if (response is not null)
                {
                    var floatMemory = response.Value.ToFloats();
                    //store them into unum search index
                    index.Add((ulong)question.IdConsulta, floatMemory.ToArray());
                    Console.WriteLine($"Inserted {question.IdConsulta} into vector database");
                }
            }

            //save from memory to file
            index.Save(vectorDatabase);
            Console.WriteLine($"Finished Creating Vector Database");
        }
        else
        {
            //load from file to memory
            index = new USearchIndex(vectorDatabase);
        }


        //system prompt used to define behavior for introduction and answers
        string system = "Eres una bot de soporte tecnico que responde solo con la informacion que se mande en tu contexto y para cualquier duda que tu tengas darle la informacion de contacto al cliente: cosmez@gmail.com numero telefonico: 445577";


        //we stream the initial introduction
        await foreach (var aiResponse in client.CompleteChatStreamingAsync(
                           [new UserChatMessage("Presentate ante el cliente de soporte tecnico, trabajas en universidad de Cosme Zamudio."), new SystemChatMessage(system)]))
        {
            if (aiResponse.ContentUpdate.Count > 0)
            {
                Console.Write(aiResponse.ContentUpdate[0].Text);
            }
        }
        Console.WriteLine(); 



        //now we ask a question to the customer
        string? line = Console.ReadLine();
        if (line is not null)
        {
            //we grab the embeddings for that question
            var response = await embeddingClient.GenerateEmbeddingAsync(line);
            if (response is not null)
            {
                var floats = response.Value.ToFloats().ToArray();
                int maxResponses = 10;
                //perform a similarity search with our vector database
                index.Search(floats, maxResponses, out var keys, out var distances);

                //with the 10 closest answers we build a prompt to send to openai
                var sbPrompt = new StringBuilder();
                sbPrompt.AppendLine($"Con las siguientes preguntas y respuestas:");
                for (int i = 0; i < maxResponses; i++)
                {
                    Console.WriteLine($"{keys[i]}\t{distances[i]}");
                    int idConsulta = (int)keys[i];
                    var question = questions.FirstOrDefault(q => q.IdConsulta == idConsulta);
                    //create the prompt with the combination of question and answer
                    if (question is not null)
                    {
                        sbPrompt.AppendLine($"Pregunta: {question.Consulta}");
                        sbPrompt.AppendLine($"Respuesta:\n {question.Texto}");
                        sbPrompt.AppendLine();
                    }
                }
                //append the initial question from the user at the end of the prompt
                sbPrompt.AppendLine($"Response la siguiente pregunta: {line}");
                string prompt = sbPrompt.ToString();
                

                //finally we call OpenAI to perform the answer
                ChatMessage[] chatMessages = [new UserChatMessage(prompt), new SystemChatMessage(system)];
                await foreach (var aiResponse in client.CompleteChatStreamingAsync(chatMessages))
                {
                    if (aiResponse.ContentUpdate.Count > 0)
                    {
                        Console.Write(aiResponse.ContentUpdate[0].Text);
                    }
                }
            }
        }
         

        index.Dispose();

    }
}


/// <summary>
/// POCO mapping the contents of our Q&A database
/// </summary>
public class Questions
{
    public required int IdConsulta { get; set; }
    public required string Consulta { get; set; }
    public required int Articulo { get; set; }
    public required string Texto { get; set; }
}
