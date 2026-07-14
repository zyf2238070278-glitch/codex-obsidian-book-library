import Foundation

enum RecognizedTextBudgetError: Error, Equatable, CustomStringConvertible {
    case limitExceeded

    var description: String {
        switch self {
        case .limitExceeded:
            return "recognized text exceeds 100,000 Unicode scalars or 400,000 UTF-8 bytes"
        }
    }
}

struct RecognizedTextBudget {
    let maximumUnicodeScalars: Int
    let maximumUTF8Bytes: Int
    private(set) var unicodeScalarCount = 0
    private(set) var utf8ByteCount = 0

    init(
        maximumUnicodeScalars: Int = 100_000,
        maximumUTF8Bytes: Int = 400_000
    ) {
        precondition(maximumUnicodeScalars >= 0)
        precondition(maximumUTF8Bytes >= 0)
        self.maximumUnicodeScalars = maximumUnicodeScalars
        self.maximumUTF8Bytes = maximumUTF8Bytes
    }

    mutating func add(_ text: String) throws {
        let (nextUnicodeScalars, scalarOverflow) = unicodeScalarCount
            .addingReportingOverflow(text.unicodeScalars.count)
        let (nextUTF8Bytes, byteOverflow) = utf8ByteCount
            .addingReportingOverflow(text.utf8.count)
        guard !scalarOverflow, !byteOverflow,
              nextUnicodeScalars <= maximumUnicodeScalars,
              nextUTF8Bytes <= maximumUTF8Bytes else {
            throw RecognizedTextBudgetError.limitExceeded
        }
        unicodeScalarCount = nextUnicodeScalars
        utf8ByteCount = nextUTF8Bytes
    }
}
