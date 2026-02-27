import Foundation

internal enum RecognitionPromptMapper {
    internal static func prompt(for task: OCRTask, prompts: GlmOCRPromptConfig) -> String {
        switch task {
        case .text:
            return prompts.textPrompt
        case .table:
            return prompts.tablePrompt
        case .formula:
            return prompts.formulaPrompt
        }
    }
}
