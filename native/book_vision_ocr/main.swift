import CoreGraphics
import CoreFoundation
import Darwin
import Foundation
import ImageIO
import Vision

private let schemaVersion = 1
private let helperVersion = "0.1.0"
private let maximumRecognizedUnicodeScalars = 100_000
private let maximumRecognizedUTF8Bytes = 400_000
private let maximumImageFileBytes = 64 * 1024 * 1024
private let maximumImageDimension = 12_000
private let maximumImagePixels = 40_000_000

private enum HelperError: Error, CustomStringConvertible {
    case invalidArguments(String)
    case invalidImage(String)
    case unsupportedLanguage(String)
    case recognition(String)

    var description: String {
        switch self {
        case .invalidArguments(let message), .invalidImage(let message),
             .unsupportedLanguage(let message), .recognition(let message):
            return message
        }
    }
}

private struct VersionPayload: Encodable {
    let schemaVersion: Int
    let version: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case version
    }
}

private struct CapabilitiesPayload: Encodable {
    let schemaVersion: Int
    let languages: [String]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case languages
    }
}

private struct BoxPayload: Encodable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

private struct LinePayload: Encodable {
    let text: String
    let confidence: Double
    let box: BoxPayload
}

private struct OCRPayload: Encodable {
    let schemaVersion: Int
    let lines: [LinePayload]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case lines
    }
}

private struct IndexedLine {
    let index: Int
    let payload: LinePayload
}

private enum Operation {
    case version
    case capabilities
    case recognize(imagePath: String, languages: [String])
}

private func emitJSON<T: Encodable>(_ payload: T) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    var data = try encoder.encode(payload)
    data.append(0x0A)
    try FileHandle.standardOutput.write(contentsOf: data)
}

private func fail(_ message: String) -> Never {
    let rendered = "book-vision-ocr: \(message)\n"
    if let data = rendered.data(using: .utf8) {
        try? FileHandle.standardError.write(contentsOf: data)
    }
    exit(2)
}

private func parseLanguages(_ raw: String) throws -> [String] {
    guard !raw.isEmpty, raw.utf8.count <= 512 else {
        throw HelperError.invalidArguments(
            "--languages must not be empty or oversized"
        )
    }
    let languages = raw
        .split(separator: ",", omittingEmptySubsequences: false)
        .map(String.init)
    guard !languages.isEmpty, languages.count <= 32,
          languages.allSatisfy({ language in
              !language.isEmpty
                  && language == language.trimmingCharacters(
                      in: .whitespacesAndNewlines
                  )
          }) else {
        throw HelperError.invalidArguments(
            "--languages must be a comma-separated list of nonblank identifiers"
        )
    }
    guard Set(languages).count == languages.count else {
        throw HelperError.invalidArguments("--languages must not contain duplicates")
    }
    return languages
}

private func parseArguments(_ arguments: [String]) throws -> Operation {
    if arguments == ["--version"] {
        return .version
    }
    if arguments == ["--capabilities"] {
        return .capabilities
    }
    guard !arguments.isEmpty else {
        throw HelperError.invalidArguments(
            "expected --version, --capabilities, or --image/--languages"
        )
    }

    var imagePath: String?
    var languages: [String]?
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard !flag.isEmpty else {
            throw HelperError.invalidArguments("empty arguments are not allowed")
        }
        guard flag == "--image" || flag == "--languages" else {
            throw HelperError.invalidArguments("unknown or misplaced flag: \(flag)")
        }
        guard index + 1 < arguments.count, !arguments[index + 1].isEmpty,
              !arguments[index + 1].hasPrefix("--") else {
            throw HelperError.invalidArguments("missing value for \(flag)")
        }
        let value = arguments[index + 1]
        if flag == "--image" {
            guard imagePath == nil else {
                throw HelperError.invalidArguments("duplicate --image flag")
            }
            imagePath = value
        } else {
            guard languages == nil else {
                throw HelperError.invalidArguments("duplicate --languages flag")
            }
            languages = try parseLanguages(value)
        }
        index += 2
    }
    guard let imagePath, let languages else {
        throw HelperError.invalidArguments("both --image and --languages are required")
    }
    return .recognize(imagePath: imagePath, languages: languages)
}

private func fileMetadataMatches(_ left: stat, _ right: stat) -> Bool {
    return left.st_dev == right.st_dev
        && left.st_ino == right.st_ino
        && left.st_size == right.st_size
        && left.st_mtimespec.tv_sec == right.st_mtimespec.tv_sec
        && left.st_mtimespec.tv_nsec == right.st_mtimespec.tv_nsec
        && left.st_ctimespec.tv_sec == right.st_ctimespec.tv_sec
        && left.st_ctimespec.tv_nsec == right.st_ctimespec.tv_nsec
}

private func readImageData(at path: String) throws -> Data {
    guard !path.isEmpty, path.utf8.count <= 32_768,
          NSString(string: path).isAbsolutePath else {
        throw HelperError.invalidImage("--image must be an absolute filesystem path")
    }
    if URL(string: path)?.scheme != nil {
        throw HelperError.invalidImage("--image must be a filesystem path, not a URL")
    }

    let descriptor = path.withCString { pointer in
        Darwin.open(pointer, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
    }
    guard descriptor >= 0 else {
        if errno == ELOOP {
            throw HelperError.invalidImage("image must not be a symbolic link")
        }
        throw HelperError.invalidImage("image could not be opened safely")
    }
    defer {
        Darwin.close(descriptor)
    }

    var fileInformation = stat()
    guard fstat(descriptor, &fileInformation) == 0 else {
        throw HelperError.invalidImage("opened image could not be inspected")
    }
    let fileType = fileInformation.st_mode & mode_t(S_IFMT)
    guard fileType == mode_t(S_IFREG) else {
        throw HelperError.invalidImage("image must be a regular file")
    }
    guard fileInformation.st_size > 0,
          fileInformation.st_size <= off_t(maximumImageFileBytes) else {
        throw HelperError.invalidImage(
            "image file size must be between 1 byte and 64 MiB"
        )
    }

    let expectedSize = Int(fileInformation.st_size)
    var data = Data(count: expectedSize)
    try data.withUnsafeMutableBytes { rawBuffer in
        guard let baseAddress = rawBuffer.baseAddress else {
            throw HelperError.invalidImage("image data buffer could not be allocated")
        }
        var offset = 0
        while offset < expectedSize {
            let count = Darwin.read(
                descriptor,
                baseAddress.advanced(by: offset),
                expectedSize - offset
            )
            if count < 0 {
                if errno == EINTR {
                    continue
                }
                throw HelperError.invalidImage("opened image could not be read")
            }
            guard count > 0 else {
                throw HelperError.invalidImage("image changed while it was being read")
            }
            offset += count
        }
    }

    var extraByte: UInt8 = 0
    while true {
        let extraCount = Darwin.read(descriptor, &extraByte, 1)
        if extraCount < 0 && errno == EINTR {
            continue
        }
        guard extraCount == 0 else {
            if extraCount < 0 {
                throw HelperError.invalidImage("opened image could not be read")
            }
            throw HelperError.invalidImage("image changed while it was being read")
        }
        break
    }
    var finalInformation = stat()
    guard fstat(descriptor, &finalInformation) == 0,
          fileMetadataMatches(fileInformation, finalInformation) else {
        throw HelperError.invalidImage("image changed while it was being read")
    }
    return data
}

private func imageDimension(
    _ rawValue: Any?,
    name: String
) throws -> UInt64 {
    guard let rawValue else {
        throw HelperError.invalidImage("image source is missing \(name)")
    }
    let coreValue = rawValue as CFTypeRef
    guard CFGetTypeID(coreValue) != CFBooleanGetTypeID(),
          let number = rawValue as? NSNumber else {
        throw HelperError.invalidImage("image \(name) must be a native number")
    }
    let value = number.doubleValue
    guard value.isFinite, value >= 1, value.rounded(.towardZero) == value else {
        throw HelperError.invalidImage(
            "image \(name) must be a finite positive integer"
        )
    }
    guard value <= Double(maximumImageDimension) else {
        throw HelperError.invalidImage(
            "image dimension exceeds the 12,000 pixel side limit"
        )
    }
    return UInt64(value)
}

private func validateImageDimensions(
    width: UInt64,
    height: UInt64
) throws {
    guard width <= UInt64(maximumImageDimension),
          height <= UInt64(maximumImageDimension) else {
        throw HelperError.invalidImage(
            "image dimension exceeds the 12,000 pixel side limit"
        )
    }
    let (pixels, overflow) = width.multipliedReportingOverflow(by: height)
    guard !overflow, pixels <= UInt64(maximumImagePixels) else {
        throw HelperError.invalidImage(
            "image pixel count exceeds the 40,000,000 pixel limit"
        )
    }
}

private func validatedImage(at path: String) throws -> CGImage {
    let data = try readImageData(at: path)
    guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
        throw HelperError.invalidImage("image source could not be created")
    }
    guard let properties = CGImageSourceCopyPropertiesAtIndex(
        source,
        0,
        nil
    ) as? [CFString: Any] else {
        throw HelperError.invalidImage("image source properties are malformed")
    }
    let width = try imageDimension(
        properties[kCGImagePropertyPixelWidth],
        name: "pixel width"
    )
    let height = try imageDimension(
        properties[kCGImagePropertyPixelHeight],
        name: "pixel height"
    )
    try validateImageDimensions(width: width, height: height)

    guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw HelperError.invalidImage("image could not be decoded")
    }
    try validateImageDimensions(
        width: UInt64(image.width),
        height: UInt64(image.height)
    )
    return image
}

private func supportedLanguages() throws -> [String] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    do {
        return try request.supportedRecognitionLanguages().sorted()
    } catch {
        throw HelperError.recognition(
            "could not query supported recognition languages: \(error.localizedDescription)"
        )
    }
}

private func normalizedBox(_ rectangle: CGRect) throws -> BoxPayload {
    var x = Double(rectangle.origin.x)
    var y = Double(rectangle.origin.y)
    var width = Double(rectangle.size.width)
    var height = Double(rectangle.size.height)
    let values = [x, y, width, height]
    guard values.allSatisfy({ $0.isFinite }), width > 0, height > 0,
          x >= -0.000_001, y >= -0.000_001,
          x <= 1.000_001, y <= 1.000_001,
          x + width <= 1.000_001, y + height <= 1.000_001 else {
        throw HelperError.recognition(
            "Vision returned an invalid normalized bounding box"
        )
    }
    x = min(max(x, 0), 1)
    y = min(max(y, 0), 1)
    width = min(width, 1 - x)
    height = min(height, 1 - y)
    guard width > 0, height > 0 else {
        throw HelperError.recognition("Vision returned an empty normalized bounding box")
    }
    return BoxPayload(x: x, y: y, width: width, height: height)
}

private func recognize(imagePath: String, languages: [String]) throws -> OCRPayload {
    let image = try validatedImage(at: imagePath)
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    let availableLanguages: [String]
    do {
        availableLanguages = try request.supportedRecognitionLanguages()
    } catch {
        throw HelperError.recognition(
            "could not query supported recognition languages: \(error.localizedDescription)"
        )
    }
    let availableSet = Set(availableLanguages)
    for language in languages where !availableSet.contains(language) {
        throw HelperError.unsupportedLanguage("unsupported recognition language: \(language)")
    }
    request.recognitionLanguages = languages

    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
    } catch {
        throw HelperError.recognition("Vision recognition failed: \(error.localizedDescription)")
    }

    var totalUnicodeScalars = 0
    var totalUTF8Bytes = 0
    var indexedLines: [IndexedLine] = []
    for (index, observation) in (request.results ?? []).enumerated() {
        guard let candidate = observation.topCandidates(1).first else {
            continue
        }
        let text = candidate.string
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            continue
        }
        let (unicodeScalars, scalarOverflow) = totalUnicodeScalars
            .addingReportingOverflow(text.unicodeScalars.count)
        let (utf8Bytes, byteOverflow) = totalUTF8Bytes
            .addingReportingOverflow(text.utf8.count)
        guard !scalarOverflow, !byteOverflow,
              unicodeScalars <= maximumRecognizedUnicodeScalars,
              utf8Bytes <= maximumRecognizedUTF8Bytes else {
            throw HelperError.recognition(
                "recognized text exceeds 100,000 Unicode scalars or 400,000 UTF-8 bytes"
            )
        }
        totalUnicodeScalars = unicodeScalars
        totalUTF8Bytes = utf8Bytes
        let confidence = Double(candidate.confidence)
        guard confidence.isFinite, confidence >= 0, confidence <= 1 else {
            throw HelperError.recognition("Vision returned invalid confidence")
        }
        let line = LinePayload(
            text: text,
            confidence: confidence,
            box: try normalizedBox(observation.boundingBox)
        )
        indexedLines.append(IndexedLine(index: index, payload: line))
    }
    indexedLines.sort { left, right in
        if left.payload.box.y != right.payload.box.y {
            return left.payload.box.y > right.payload.box.y
        }
        if left.payload.box.x != right.payload.box.x {
            return left.payload.box.x < right.payload.box.x
        }
        if left.payload.text != right.payload.text {
            return left.payload.text < right.payload.text
        }
        return left.index < right.index
    }
    return OCRPayload(
        schemaVersion: schemaVersion,
        lines: indexedLines.map(\.payload)
    )
}

do {
    let operation = try parseArguments(Array(CommandLine.arguments.dropFirst()))
    switch operation {
    case .version:
        try emitJSON(
            VersionPayload(schemaVersion: schemaVersion, version: helperVersion)
        )
    case .capabilities:
        try emitJSON(
            CapabilitiesPayload(
                schemaVersion: schemaVersion,
                languages: try supportedLanguages()
            )
        )
    case .recognize(let imagePath, let languages):
        try emitJSON(try recognize(imagePath: imagePath, languages: languages))
    }
} catch let error as HelperError {
    fail(error.description)
} catch {
    fail("unexpected failure: \(error.localizedDescription)")
}
