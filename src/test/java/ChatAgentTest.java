import net.echo.brain4j.nlp.AlphabetInitialization;
import net.echo.brain4j.nlp.LabelTransformer;
import net.echo.brain4j.nlp.agents.attention.AttentionMechanism;
import net.echo.brain4j.nlp.agents.attention.impl.SelfAttention;
import net.echo.brain4j.nlp.agents.impl.ChatAgent;
import net.echo.brain4j.training.data.nlp.ConversationDataSet;

public class ChatAgentTest {
    public static void main(String[] args) {
        AttentionMechanism selfAttention = new SelfAttention(512, 0.6, 0.95);
        ChatAgent agent = new ChatAgent(selfAttention, 512, 512, 0.6, 0.95);
        LabelTransformer transformer = new LabelTransformer(AlphabetInitialization.NORMAL);
        ConversationDataSet trainingData = new ConversationDataSet(512, transformer,
                "Hello, how are you?",
                "I'm doing great, thanks for asking!",
                "What's the weather like?",
                "It's sunny and warm today."
        );

        System.out.println("Starting training with max 100 epochs...");
        try {
            agent.train(trainingData);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("Training completed");

        String userInput = "Hello, how are you?";
        System.out.println("\nUser: " + userInput);
        String response = agent.generateResponse(userInput);
        System.out.println("ChatBot: " + response);
        System.out.println(agent.getModel().getStats());
    }
}


